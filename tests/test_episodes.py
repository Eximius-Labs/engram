"""Phase E (A + B): episode change-point segmentation and salience scoring.

Deterministic, GPU-free. Rows are built with hand-controlled unit vectors so the change-point
and novelty maths are asserted exactly, independent of any embedder.
"""

import numpy as np

from robomem.episodes import DEFAULT_EPISODE_THRESHOLD, Episode, assign_episodes

DIM = 16


def _vec(base: int, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[base] = 1.0
    if jitter:
        v = v + jitter * np.random.RandomState(seed).randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _row(eid, t, base, jitter=0.0, seed=0, modality="motion", source="imu0", label=None):
    meta = {"ground_truth_label": label} if label else {}
    return {"event_id": eid, "t_start": float(t), "t_end": float(t) + 1.0,
            "modality": modality, "source": source, "vector": _vec(base, jitter, seed),
            "meta": meta}


def _calm_then_shake_stream():
    """One motion stream: two near-identical calm windows, then a very different shake window."""
    return [
        _row("m0", 0.0, base=0, jitter=0.02, seed=1),
        _row("m1", 1.0, base=0, jitter=0.02, seed=2),
        _row("m2", 2.0, base=1),   # orthogonal -> a change point
    ]


# --------------------------------------------------------------- segmentation
def test_contiguous_similar_windows_form_one_episode():
    rows = [_row("m0", 0.0, base=0, jitter=0.02, seed=1),
            _row("m1", 1.0, base=0, jitter=0.02, seed=2)]
    ep_by_event, _, episodes = assign_episodes(rows)
    assert len(episodes) == 1
    assert ep_by_event["m0"] == ep_by_event["m1"]
    assert episodes[0].member_ids == ["m0", "m1"]


def test_a_content_jump_opens_a_new_episode():
    ep_by_event, _, episodes = assign_episodes(_calm_then_shake_stream())
    assert len(episodes) == 2
    assert ep_by_event["m0"] == ep_by_event["m1"] != ep_by_event["m2"]
    # episodes are time-ordered and span their members
    assert episodes[0].t_start == 0.0 and episodes[0].t_end == 2.0
    assert episodes[1].t_start == 2.0


def test_episode_records_are_complete():
    _, _, episodes = assign_episodes(_calm_then_shake_stream())
    ep = episodes[0]
    assert isinstance(ep, Episode)
    assert ep.modality == "motion" and ep.source == "imu0"
    assert ep.centroid is not None and ep.centroid.shape == (DIM,)
    assert abs(np.linalg.norm(ep.centroid) - 1.0) < 1e-5
    d = ep.as_dict()
    assert d["n_members"] == 2 and 0.0 <= d["salience"] <= 1.0


def test_threshold_is_tunable():
    rows = _calm_then_shake_stream()
    # a threshold above the jump distance keeps everything in one episode
    _, _, loose = assign_episodes(rows, threshold=1.5)
    assert len(loose) == 1
    # a tiny threshold splits even the small jitter between the two calm windows
    _, _, tight = assign_episodes(rows, threshold=0.001)
    assert len(tight) == 3


def test_streams_are_segmented_independently():
    rows = _calm_then_shake_stream() + [
        _row("a0", 0.5, base=5, modality="audio", source="mic0"),
        _row("a1", 1.5, base=6, modality="audio", source="mic0"),  # jump -> 2 audio episodes
    ]
    _, _, episodes = assign_episodes(rows)
    mods = sorted({e.modality for e in episodes})
    assert mods == ["audio", "motion"]
    # 2 motion episodes + 2 audio episodes, no episode mixes modalities
    assert len(episodes) == 4
    for e in episodes:
        assert len({r for r in e.member_ids}) == len(e.member_ids)


def test_pairwise_method_runs_and_segments():
    _, _, episodes = assign_episodes(_calm_then_shake_stream(), method="pairwise")
    assert len(episodes) == 2


# -------------------------------------------------------------------- salience
def test_first_window_has_zero_salience_and_a_shake_is_most_salient():
    _, sal, _ = assign_episodes(_calm_then_shake_stream())
    assert sal["m0"] == 0.0                 # first window: no prior expectation
    assert 0.0 <= sal["m1"] <= 1.0 and 0.0 <= sal["m2"] <= 1.0
    # the shake is far more novel than the second calm window
    assert sal["m2"] > sal["m1"]
    assert sal["m2"] == 1.0                 # peak novelty normalizes to 1.0


def test_salience_is_bounded_and_populates_episode_peak():
    _, sal, episodes = assign_episodes(_calm_then_shake_stream())
    assert all(0.0 <= v <= 1.0 for v in sal.values())
    shake_ep = [e for e in episodes if "m2" in e.member_ids][0]
    assert shake_ep.salience == sal["m2"]


def test_default_threshold_constant_is_reasonable():
    assert 0.0 < DEFAULT_EPISODE_THRESHOLD < 1.0


# ----------------------------------------------------------- temporal gap break
def test_max_gap_splits_similar_windows_far_apart_in_time():
    # two near-identical windows 40 s apart: without a gap break they merge; with one they split.
    rows = [_row("m0", 0.0, base=0, jitter=0.01, seed=1),
            _row("m1", 40.0, base=0, jitter=0.01, seed=2)]
    _, _, merged = assign_episodes(rows, threshold=0.35)
    assert len(merged) == 1                      # embedding-similar -> one episode by content
    _, _, split = assign_episodes(rows, threshold=0.35, max_gap=2.0)
    assert len(split) == 2                        # 40 s > 2 s gap -> two distinct occurrences
    assert split[0].t_start == 0.0 and split[1].t_start == 40.0


def test_max_gap_leaves_contiguous_windows_untouched():
    # back-to-back similar windows (no gap) stay one episode even with max_gap set.
    rows = [_row("m0", 0.0, base=0, jitter=0.01, seed=1),
            _row("m1", 1.0, base=0, jitter=0.01, seed=2)]
    _, _, eps = assign_episodes(rows, threshold=0.35, max_gap=2.0)
    assert len(eps) == 1
