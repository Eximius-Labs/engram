"""GPU-free tests for the session builder: stitched timeline math, event shape, and that the
emitted manifest drives build_rows(FakeEmbedder, ...) to the rows the demo will query.
"""

import numpy as np

from robomem import FakeEmbedder, RobotMemory
from robomem.ingest import build_rows
from robomem.session_builder import (
    Clip,
    Segment,
    hit_segment_index,
    place_clips,
    segment_for_time,
    stitch_session,
    window_events,
)


def demo_clips():
    """Four visually + sonically distinct, nameable clips (the demo's shape)."""
    return [
        Clip(path="vgg/dog_barking.mp4", label="a dog barking", duration=10.0),
        Clip(path="vgg/acoustic_guitar.mp4", label="playing acoustic guitar", duration=10.0),
        Clip(path="vgg/motorcycle.mp4", label="a motorcycle engine revving", duration=10.0),
        Clip(path="vgg/person_shouting.mp4", label="a person shouting", duration=10.0),
    ]


# --------------------------------------------------------------- timeline math
def test_place_clips_cumulative_offsets_no_gap():
    segs = place_clips(demo_clips(), gap=0.0)
    assert [s.t_start for s in segs] == [0.0, 10.0, 20.0, 30.0]
    assert [s.t_end for s in segs] == [10.0, 20.0, 30.0, 40.0]
    assert [s.index for s in segs] == [0, 1, 2, 3]
    assert [s.label for s in segs][2] == "a motorcycle engine revving"


def test_place_clips_inserts_gap():
    segs = place_clips(demo_clips(), gap=2.0)
    assert [s.t_start for s in segs] == [0.0, 12.0, 24.0, 36.0]
    assert [s.t_end for s in segs] == [10.0, 22.0, 34.0, 46.0]


def test_place_clips_rejects_nonpositive_duration():
    try:
        place_clips([Clip(path="x", label="x", duration=0.0)])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on zero duration")


def test_segment_midpoint_and_contains():
    segs = place_clips(demo_clips())
    assert segs[0].mid == 5.0
    assert segs[1].contains(15.0)
    assert not segs[1].contains(25.0)
    assert segment_for_time(segs, 25.0).index == 2
    assert segment_for_time(segs, 999.0) is None


# ----------------------------------------------------------------- event shape
def test_stitch_emits_image_and_audio_per_segment():
    events, segs = stitch_session(demo_clips())
    assert len(segs) == 4
    imgs = [e for e in events if e["modality"] == "image"]
    auds = [e for e in events if e["modality"] == "audio"]
    assert len(imgs) == 4 and len(auds) == 4
    # image is a mid-frame instant; audio spans the whole segment window
    assert imgs[2]["t_start"] == imgs[2]["t_end"] == 25.0
    assert auds[2]["t_start"] == 20.0 and auds[2]["t_end"] == 30.0
    # ground-truth label + segment index ride along on every event
    for e in events:
        assert e["meta"]["ground_truth_label"] in {c.label for c in demo_clips()}
        assert "segment_index" in e["meta"]


def test_stitch_video_opt_in():
    no_vid, _ = stitch_session(demo_clips())
    assert not any(e["modality"] == "video" for e in no_vid)
    with_vid, _ = stitch_session(demo_clips(), emit_video=True)
    vids = [e for e in with_vid if e["modality"] == "video"]
    assert len(vids) == 4
    assert vids[0]["t_start"] == 0.0 and vids[0]["t_end"] == 10.0


def test_stitch_sources_and_refs():
    events, _ = stitch_session(demo_clips(), emit_video=True)
    by_mod = {}
    for e in events:
        by_mod.setdefault(e["modality"], e)
    assert by_mod["image"]["source"] == "cam0"
    assert by_mod["audio"]["source"] == "mic0"
    assert by_mod["video"]["source"] == "cam0_clip"
    # audio_path/video_path override falls back to path when unset
    assert by_mod["audio"]["path_or_data"] == "vgg/dog_barking.mp4"


def test_stitch_respects_audio_video_path_overrides():
    clips = [Clip(path="v.mp4", label="x", duration=4.0,
                  audio_path="a.wav", video_path="v2.mp4")]
    events, _ = stitch_session(clips, emit_video=True)
    refs = {e["modality"]: e["path_or_data"] for e in events}
    assert refs["image"] == "v.mp4"
    assert refs["audio"] == "a.wav"
    assert refs["video"] == "v2.mp4"


def test_stitch_rejects_empty():
    try:
        stitch_session([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on empty clip list")


# ------------------------------------------------------------- window builder
def test_window_events_nonoverlap_clamps_last():
    evs = window_events(25.0, modality="audio", path="m.wav", window=10.0, source="mic0")
    assert [(e["t_start"], e["t_end"]) for e in evs] == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    assert evs[-1]["meta"]["window_index"] == 2


def test_window_events_drop_partial_and_stride():
    evs = window_events(25.0, modality="audio", path="m.wav", window=10.0, stride=5.0,
                        drop_last_partial=True)
    # starts 0,5,10,15 give full windows; 20 would end at 30 (partial) -> dropped
    assert [e["t_start"] for e in evs] == [0.0, 5.0, 10.0, 15.0]
    assert all(e["duration"] == 10.0 for e in evs)


def test_window_events_offset_and_label():
    evs = window_events(6.0, modality="video", path="c.mp4", window=3.0, t0=100.0,
                        label="segmentX")
    assert evs[0]["t_start"] == 100.0 and evs[1]["t_end"] == 106.0
    assert evs[0]["meta"]["ground_truth_label"] == "segmentX"


# ------------------------------------------------ hit-to-segment attribution
def test_hit_segment_index_prefers_meta_then_time():
    _, segs = stitch_session(demo_clips())

    class _M:
        def __init__(self, meta, t_start, t_end):
            self.meta, self.t_start, self.t_end = meta, t_start, t_end

    # meta wins
    assert hit_segment_index(segs, _M({"segment_index": 3}, 0.0, 0.0)) == 3
    # falls back to timestamp when no meta index
    assert hit_segment_index(segs, _M({}, 24.0, 26.0)) == 2
    # out of range -> None
    assert hit_segment_index(segs, _M({}, 999.0, 1000.0)) is None


# ------------------------------------------ end-to-end through build_rows + recall
def test_build_rows_over_stitched_manifest(tmp_path):
    events, segs = stitch_session(demo_clips())
    emb = FakeEmbedder()
    rows, stats = build_rows(emb, events, dedup_tau=None)
    # 4 image + 4 audio events, all kept (no duplicates), one row each
    assert stats.embedded == 8 and stats.kept == 8 and stats.deduped == 0
    assert {r["modality"] for r in rows} == {"image", "audio"}
    assert len(rows) == 8
    # every row's vector is full-width and carries its ground-truth label
    import json
    for r in rows:
        assert r["vector"].shape == (emb.dim,)
        assert json.loads(r["meta"])["ground_truth_label"]


def test_stitched_session_recall_hits_right_segment(tmp_path):
    """The FakeEmbedder keys on filename concept tokens, so a text query for a segment's
    concept must recall that segment's window — the same discrimination the Modal demo asserts,
    proven here GPU-free."""
    events, segs = stitch_session(demo_clips())
    mem = RobotMemory.open(str(tmp_path / "sess"), embedder=FakeEmbedder())
    mem.index(events, dedup_tau=None)

    # visual query -> image gallery, must land in the motorcycle segment (index 2)
    hits = mem.recall("motorcycle engine", modality="image", k=4)
    assert hits, "no image hits"
    assert hit_segment_index(segs, hits[0]) == 2

    # audio query -> audio gallery, a DIFFERENT segment (guitar, index 1)
    hits_a = mem.recall("acoustic guitar", modality="audio", k=4)
    assert hits_a, "no audio hits"
    assert hit_segment_index(segs, hits_a[0]) == 1

    # the two resolve to different segments -> content+time discrimination, not luck
    assert hit_segment_index(segs, hits[0]) != hit_segment_index(segs, hits_a[0])
