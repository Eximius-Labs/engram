"""Build a timestamped session manifest from source media.

A *session manifest* is the list of timestamped events that :func:`robomem.ingest.build_rows`
consumes (see the module docstring there for the event shape). This builder produces that list
at the reference level: paths + absolute timestamps + meta. It never decodes a frame, never
loads a model, and imports neither torch nor fusion_embedding, so it stays GPU-free and
independent of the embedding stack. Actual frame / audio decoding for embedding happens on the
side that owns the model (the Modal demo), which reads the ``payload`` reference each event
carries.

Two builders:

* :func:`stitch_session` — lay an ordered list of clips end-to-end on ONE session timeline at
  known cumulative offsets, and emit per-clip events (image mid-frame reference, audio window,
  optional video-clip event) with correct absolute ``t_start`` / ``t_end`` and a
  ``meta.ground_truth_label``. This is what makes scripted recall queries verifiable: every
  emitted event knows which stitched segment it belongs to.

* :func:`window_events` — given one media's duration, emit fixed-window events over it (the
  generic sliding-window builder a real log reader would use).

Both return plain dict events; the caller passes them straight to ``build_rows`` /
``RobotMemory.index``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Modalities this builder emits references for. Kept in sync with robomem.ingest._MODALITIES
# but not imported from there (ingest imports numpy; the builder stays dependency-free).
_IMAGE = "image"
_AUDIO = "audio"
_VIDEO = "video"


@dataclass
class Clip:
    """One source clip to place on the stitched timeline.

    ``path`` is the media reference the Modal side will decode (a file path, a volume path, or
    any opaque token the embedder's decode step understands — the builder never opens it).
    ``label`` is the human-readable ground-truth content label ("a dog barking"). ``duration``
    is the clip length in seconds. ``audio_path`` / ``video_path`` default to ``path`` when the
    one file carries every modality (an mp4 with sound); override when audio lives elsewhere.
    """

    path: str
    label: str
    duration: float
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def image_ref(self) -> str:
        return self.path

    def audio_ref(self) -> str:
        return self.audio_path or self.path

    def video_ref(self) -> str:
        return self.video_path or self.path


@dataclass
class Segment:
    """A placed clip: where it lives on the session timeline."""

    index: int
    label: str
    t_start: float
    t_end: float
    clip: Clip

    @property
    def mid(self) -> float:
        return round((self.t_start + self.t_end) / 2.0, 6)

    def contains(self, t: float, *, slack: float = 0.0) -> bool:
        return (self.t_start - slack) <= t <= (self.t_end + slack)


def place_clips(clips, gap: float = 0.0) -> list[Segment]:
    """Lay ``clips`` end-to-end from t=0, inserting ``gap`` seconds of silence between them.

    Returns the :class:`Segment` list with absolute cumulative offsets. Pure arithmetic; this is
    the single source of truth for where each clip sits, so tests assert against it directly.
    """
    segments: list[Segment] = []
    t = 0.0
    for i, clip in enumerate(clips):
        dur = float(clip.duration)
        if dur <= 0:
            raise ValueError(f"clip {i} ({clip.label!r}) has non-positive duration {dur}")
        seg = Segment(index=i, label=clip.label, t_start=round(t, 6),
                      t_end=round(t + dur, 6), clip=clip)
        segments.append(seg)
        t = seg.t_end + gap
    return segments


def _event(modality, ref, t_start, t_end, source, segment, extra_meta=None):
    meta = {
        "ground_truth_label": segment.label,
        "segment_index": segment.index,
        "segment_t_start": segment.t_start,
        "segment_t_end": segment.t_end,
        "path": ref,
    }
    if segment.clip.meta:
        meta.update(segment.clip.meta)
    if extra_meta:
        meta.update(extra_meta)
    return {
        "id": f"seg{segment.index}_{modality}",
        "t_start": round(float(t_start), 6),
        "t_end": round(float(t_end), 6),
        "duration": round(float(t_end) - float(t_start), 6),
        "modality": modality,
        "path_or_data": ref,
        "source": source,
        "meta": meta,
    }


def stitch_session(clips, *, gap: float = 0.0,
                   emit_image: bool = True, emit_audio: bool = True,
                   emit_video: bool = False,
                   image_source: str = "cam0", audio_source: str = "mic0",
                   video_source: str = "cam0_clip") -> tuple[list[dict], list[Segment]]:
    """Stitch ``clips`` onto one timeline and emit per-segment events.

    For each placed segment we emit (subject to the ``emit_*`` flags):

    * an ``image`` event — a representative mid-frame reference, timestamped at the segment
      midpoint (a zero-duration instant, the way a sampled keyframe is);
    * an ``audio`` event — the whole segment's audio window (``t_start`` .. ``t_end``);
    * optionally a ``video`` event — the whole segment's clip window.

    Every event carries ``meta.ground_truth_label`` and ``meta.segment_index`` so a recall hit
    can be checked against the segment it should belong to. Returns ``(events, segments)`` —
    events feed ``build_rows``; segments are the ground-truth map the demo asserts against.
    """
    clips = list(clips)
    if not clips:
        raise ValueError("stitch_session needs at least one clip")
    segments = place_clips(clips, gap=gap)
    events: list[dict] = []
    for seg in segments:
        if emit_image:
            events.append(_event(_IMAGE, seg.clip.image_ref(), seg.mid, seg.mid,
                                  image_source, seg, {"frame_time": seg.mid}))
        if emit_audio:
            events.append(_event(_AUDIO, seg.clip.audio_ref(), seg.t_start, seg.t_end,
                                 audio_source, seg))
        if emit_video:
            events.append(_event(_VIDEO, seg.clip.video_ref(), seg.t_start, seg.t_end,
                                 video_source, seg))
    return events, segments


def window_events(duration: float, *, modality: str, path: str, window: float,
                  stride: Optional[float] = None, source: Optional[str] = None,
                  t0: float = 0.0, label: Optional[str] = None,
                  drop_last_partial: bool = False) -> list[dict]:
    """Emit fixed-window events over a single media's ``duration`` seconds.

    Windows start at ``t0`` and advance by ``stride`` (default: ``window``, i.e. non-overlap).
    Each event references the same ``path`` with a ``[t_start, t_end)`` window recorded in meta,
    so a decode step can seek into the source. Set ``drop_last_partial`` to discard a trailing
    window shorter than ``window``; otherwise the last window is clamped to ``duration``.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    stride = window if stride is None else float(stride)
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    duration = float(duration)
    events: list[dict] = []
    idx = 0
    start = 0.0
    while start < duration - 1e-9:
        end = start + window
        partial = end > duration + 1e-9
        if partial:
            if drop_last_partial:
                break
            end = duration
        meta = {"window_index": idx, "path": path}
        if label is not None:
            meta["ground_truth_label"] = label
        events.append({
            "id": f"{modality}_win{idx}",
            "t_start": round(t0 + start, 6),
            "t_end": round(t0 + end, 6),
            "duration": round(end - start, 6),
            "modality": modality,
            "path_or_data": path,
            "source": source,
            "meta": meta,
        })
        idx += 1
        start += stride
    return events


def segment_for_time(segments, t: float, *, slack: float = 0.0) -> Optional[Segment]:
    """Return the segment whose window contains ``t`` (with optional ``slack``), else None."""
    for seg in segments:
        if seg.contains(t, slack=slack):
            return seg
    return None


def hit_segment_index(segments, moment, *, slack: float = 1e-6) -> Optional[int]:
    """Which stitched segment does a recall :class:`~robomem.schema.Moment` (or dict) land in?

    A merged segment spans ``[t_start, t_end]``; we test its midpoint against the ground-truth
    segment windows. Prefers the event's own ``meta.segment_index`` when present (exact), and
    falls back to a timestamp lookup otherwise.
    """
    meta = getattr(moment, "meta", None)
    if meta is None and isinstance(moment, dict):
        meta = moment.get("meta")
    if isinstance(meta, dict) and meta.get("segment_index") is not None:
        return int(meta["segment_index"])
    t_start = getattr(moment, "t_start", None)
    t_end = getattr(moment, "t_end", None)
    if t_start is None and isinstance(moment, dict):
        t_start, t_end = moment.get("t_start"), moment.get("t_end")
    if t_start is None:
        return None
    mid = (float(t_start) + float(t_end if t_end is not None else t_start)) / 2.0
    seg = segment_for_time(segments, mid, slack=slack)
    return seg.index if seg is not None else None
