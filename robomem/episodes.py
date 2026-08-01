"""Episodic segmentation (Phase E, part A + B).

Turns a flat table of embedded windows into *episodes*: contiguous stretches of one stream
whose content stays coherent. An episode boundary is a **change-point** in the embedding
sequence -- when a window drifts far enough from the running centroid of the episode so far, a
new episode begins. This is deterministic and driven by a single tunable ``threshold``, so the
same session always segments the same way.

Two Phase-E schema hooks get populated from this:

* ``episode_id`` -- every Tier-0 row is stamped with the episode it belongs to.
* ``salience``   -- a per-window novelty score in ``[0, 1]``: how surprising the window is
  relative to the running expectation of its stream. A boundary window (a sudden shake after a
  calm stretch) scores high; a window that looks like everything before it scores low.

Segmentation runs **per stream** -- per ``(modality, source)`` -- because a cosine distance
between an image vector and an audio vector is dominated by the modality gap, not by a content
change. Streams are independent, so segmenting a modality-filtered subset gives the same
within-stream boundaries as segmenting the whole table; the temporal planner relies on that.

This module is pure numpy. It imports nothing from torch or fusion_embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .embedder import l2_normalize

# Cosine-distance (1 - cos) a window must exceed, versus the running centroid of the current
# episode, to open a new episode. 0.35 -> episodes break on a clear content change but tolerate
# small drift within one scene. Tunable per call.
DEFAULT_EPISODE_THRESHOLD = 0.35


@dataclass
class Episode:
    """A Tier-1 episode record: one coherent stretch of a single stream."""

    episode_id: str
    t_start: float
    t_end: float
    modality: str
    source: Optional[str]
    member_ids: list = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    label: Optional[str] = None
    salience: float = 0.0

    def as_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "modality": self.modality,
            "source": self.source,
            "member_ids": list(self.member_ids),
            "label": self.label,
            "salience": round(float(self.salience), 6),
            "n_members": len(self.member_ids),
        }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def _stream_key(row: dict) -> tuple:
    return (row["modality"], row.get("source"))


def assign_episodes(rows, threshold: float = DEFAULT_EPISODE_THRESHOLD,
                    method: str = "drift", max_gap: Optional[float] = None):
    """Segment ``rows`` into episodes and score per-window salience.

    Parameters
    ----------
    rows : list of dicts each carrying ``event_id``, ``t_start``, ``t_end``, ``modality``,
        ``source``, ``vector`` (a full-width unit vector), and optionally ``meta``.
    threshold : cosine-distance change-point threshold (see module constant).
    method : ``"drift"`` compares each window to the running centroid of the current episode
        (robust to jitter); ``"pairwise"`` compares to the immediately previous window.
    max_gap : when set, a temporal gap larger than this (seconds) between a window and the
        previous window in its stream also opens a new episode -- two similar-looking windows
        far apart in time are distinct occurrences, not one episode. ``None`` disables it.

    Returns
    -------
    (episode_by_event, salience_by_event, episodes)
        ``episode_by_event`` maps ``event_id -> episode_id``; ``salience_by_event`` maps
        ``event_id -> float in [0, 1]``; ``episodes`` is the list of :class:`Episode` records
        (time-ordered). Deterministic for a given ``(rows, threshold, method)``.
    """
    if method not in ("drift", "pairwise"):
        raise ValueError(f"unknown method {method!r}; use 'drift' or 'pairwise'")

    # Group into streams; process streams in a stable order for deterministic episode ids.
    streams: dict[tuple, list[dict]] = {}
    for r in rows:
        streams.setdefault(_stream_key(r), []).append(r)
    ordered_keys = sorted(streams.keys(), key=lambda k: (str(k[0]), str(k[1])))

    episode_by_event: dict[str, str] = {}
    raw_salience: dict[str, Optional[float]] = {}
    episodes: list[Episode] = []
    ep_counter = 0

    for key in ordered_keys:
        modality, source = key
        stream = sorted(streams[key], key=lambda e: (float(e["t_start"]), str(e["event_id"])))

        # Stream-level running expectation (never reset at a boundary) drives salience: a window
        # is "surprising" to the extent it differs from everything seen so far in the stream.
        stream_mean: Optional[np.ndarray] = None
        stream_seen = 0

        cur_members: list[dict] = []
        cur_centroid: Optional[np.ndarray] = None   # centroid of the CURRENT episode
        prev_vec: Optional[np.ndarray] = None
        prev_t_end: Optional[float] = None

        def _flush(members):
            nonlocal ep_counter
            if not members:
                return
            eid = f"ep{ep_counter:04d}"
            ep_counter += 1
            vecs = np.stack([np.asarray(m["vector"], dtype=np.float32) for m in members])
            centroid = l2_normalize(vecs.mean(axis=0))
            for m in members:
                episode_by_event[m["event_id"]] = eid
            label = _dominant_label(members)
            episodes.append(Episode(
                episode_id=eid,
                t_start=float(min(m["t_start"] for m in members)),
                t_end=float(max(m["t_end"] for m in members)),
                modality=modality, source=source,
                member_ids=[m["event_id"] for m in members],
                centroid=centroid, label=label,
            ))

        for r in stream:
            v = np.asarray(r["vector"], dtype=np.float32)

            # --- salience: novelty vs the stream's running mean (first window = no expectation)
            if stream_mean is None:
                raw_salience[r["event_id"]] = None      # baseline; normalized to 0 later
            else:
                raw_salience[r["event_id"]] = 1.0 - _cos(v, stream_mean)

            # --- change-point: does this window open a new episode?
            if cur_centroid is None:
                cur_members = [r]
                cur_centroid = v.copy()
            else:
                ref = prev_vec if method == "pairwise" else cur_centroid
                dist = 1.0 - _cos(v, ref)
                gap_break = (max_gap is not None and prev_t_end is not None
                             and float(r["t_start"]) - prev_t_end > max_gap)
                if dist > threshold or gap_break:
                    _flush(cur_members)
                    cur_members = [r]
                    cur_centroid = v.copy()
                else:
                    cur_members.append(r)
                    vecs = np.stack([np.asarray(m["vector"], dtype=np.float32)
                                     for m in cur_members])
                    cur_centroid = l2_normalize(vecs.mean(axis=0))
            prev_vec = v
            prev_t_end = float(r["t_end"])

            # update the stream running mean AFTER using it for salience
            stream_mean = v.copy() if stream_mean is None else (
                (stream_mean * stream_seen + v) / (stream_seen + 1))
            stream_seen += 1

        _flush(cur_members)

    salience_by_event = _normalize_salience(raw_salience)
    # attach the peak member salience to each episode
    for ep in episodes:
        ep.salience = max((salience_by_event[e] for e in ep.member_ids), default=0.0)
    episodes.sort(key=lambda e: (e.t_start, e.episode_id))
    return episode_by_event, salience_by_event, episodes


def _dominant_label(members) -> Optional[str]:
    """Most common ``meta.ground_truth_label`` among members (descriptive metadata, not a query
    result). Returns None when members carry no such label."""
    counts: dict[str, int] = {}
    for m in members:
        meta = m.get("meta")
        if isinstance(meta, str):
            try:
                import json
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict):
            lab = meta.get("ground_truth_label")
            if lab:
                counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _normalize_salience(raw: dict) -> dict:
    """Min-max normalize raw novelty into ``[0, 1]`` across the whole session.

    ``None`` raws (first window of each stream -- no prior expectation) map to 0.0. When every
    measured novelty is identical the signal carries no information, so all map to 0.0.
    """
    vals = [v for v in raw.values() if v is not None]
    out: dict[str, float] = {}
    if not vals:
        return {k: 0.0 for k in raw}
    lo, hi = float(min(vals)), float(max(vals))
    span = hi - lo
    for k, v in raw.items():
        if v is None:
            out[k] = 0.0
        elif span < 1e-9:
            out[k] = 0.0
        else:
            out[k] = float((v - lo) / span)
    return out
