"""Tactile as a first-class modality: ingest, recall, and temporal ops, GPU-free.

Tactile events carry a pressure window (frames of a 32x32 taxel grid) either inline as
{"data": [[...], ...]} or as a path the embedder loads itself. In production the vectors
come from the Tactus pack through UnifiedEmbedder.embed_tactile; here the FakeEmbedder
stands in, so the whole path runs with no model, no GPU, and no network.
"""
import numpy as np
import pytest

from robomem import FakeEmbedder, RobotMemory
from robomem.embedder import Embedder


def _pressure(seed: int):
    rng = np.random.RandomState(seed)
    return (rng.rand(8, 32, 32) * 255).astype(np.uint8).tolist()


def _session():
    return [
        {"id": "img_mug", "t": 1.0, "modality": "image",
         "path_or_data": "cam0/mug.png", "source": "cam0"},
        {"id": "tac_grasp_mug", "t": 1.2, "duration": 0.5, "modality": "tactile",
         "path_or_data": {"data": _pressure(0)}, "source": "hand0"},
        {"id": "tac_grasp_can", "t": 4.0, "duration": 0.5, "modality": "tactile",
         "path_or_data": {"data": _pressure(1)}, "source": "hand0"},
        {"id": "tac_path", "t": 6.0, "modality": "tactile",
         "path_or_data": "hand0/grasp_000.npy", "source": "hand0"},
        {"id": "txt_note", "t": 7.0, "modality": "text",
         "path_or_data": "operator handed the robot a mug", "source": "log"},
    ]


@pytest.fixture()
def mem(tmp_path):
    m = RobotMemory.open(str(tmp_path / "s.lancedb"), embedder=FakeEmbedder())
    m.index(_session())
    return m


def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(), Embedder)
    v = FakeEmbedder().embed_tactile(np.zeros((8, 32, 32), np.float32))
    v = np.asarray(v, np.float32).reshape(-1)
    assert np.isfinite(v).all() and abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_tactile_events_are_indexed(mem):
    rows = mem.timeline()
    tac = [r for r in rows if r.modality == "tactile"]
    assert len(tac) == 3
    assert {r.event_id for r in tac} == {"tac_grasp_mug", "tac_grasp_can", "tac_path"}


def test_recall_restricted_to_tactile(mem):
    hits = mem.recall("a firm grasp", modality="tactile", k=5)
    assert hits, "tactile recall returned nothing"
    assert all(h.modality == "tactile" for h in hits)


def test_tactile_dict_and_path_payloads_differ(mem):
    # inline windows and a path payload must embed to distinct vectors: ranking three distinct
    # tactile events against one query must produce three distinct scores (collapsed vectors
    # would tie), exercised through the public recall path.
    hits = mem.recall("a grasp", modality="tactile", k=5)
    scores = [h.score for h in hits]
    assert len(hits) == 3
    assert len({round(sc, 6) for sc in scores}) == 3, f"vectors collapsed: {scores}"


def test_temporal_last_on_tactile(mem):
    hit = mem.last("a grasp", modality="tactile")
    assert hit is not None
    # the most recent tactile event wins the recency-weighted read
    assert hit.t_start >= 4.0


def test_unknown_modality_still_rejected(tmp_path):
    m = RobotMemory.open(str(tmp_path / "x.lancedb"), embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        m.index([{"t": 0.0, "modality": "smell", "path_or_data": "?"}])
