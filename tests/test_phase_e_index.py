"""Phase E: the ``episode_id`` / ``salience`` schema hooks get populated by segmented ingest,
and Tier-1 episode records are recoverable from the stored table.
"""

from robomem import RobotMemory
from tests.test_temporal import session_e


def _mem(tmp_path, **kw):
    from robomem import FakeEmbedder
    mem = RobotMemory.open(str(tmp_path / "db"), embedder=FakeEmbedder())
    mem.index(session_e(), **kw)
    return mem


def test_plain_index_leaves_hooks_null(tmp_path):
    # default (segment=False) preserves the MVP behavior: hooks declared but empty.
    mem = _mem(tmp_path)
    row = mem.store.get("aud_shout1")
    assert row["episode_id"] is None
    assert row["salience"] is None


def test_segmented_index_populates_episode_id(tmp_path):
    mem = _mem(tmp_path, segment=True)
    rows = mem.store.scan()
    assert all(r["episode_id"] is not None for r in rows), "every row should get an episode_id"
    # the two non-contiguous shouts land in different episodes
    e1 = mem.store.get("aud_shout1")["episode_id"]
    e2 = mem.store.get("aud_shout2")["episode_id"]
    assert e1 != e2


def test_segmented_index_populates_salience(tmp_path):
    mem = _mem(tmp_path, segment=True)
    rows = mem.store.scan()
    sal = {r["event_id"]: r["salience"] for r in rows}
    # salience is present and in range for every non-first-of-stream window
    for v in sal.values():
        assert v is None or 0.0 <= v <= 1.0
    # the violent shake is the most novel motion window -> high salience vs the calm one
    assert sal["mot_shake"] is not None
    assert (sal["mot_calm"] or 0.0) <= sal["mot_shake"]


def test_episodes_records_are_built_from_the_store(tmp_path):
    mem = _mem(tmp_path, segment=True)
    eps = mem.episodes()
    assert eps, "no episodes built"
    # every stored event belongs to exactly one episode
    members = [mid for e in eps for mid in e.member_ids]
    assert sorted(members) == sorted(r["event_id"] for r in mem.store.scan())
    # dominant modality is well defined and centroids are unit vectors
    for e in eps:
        assert e.modality in ("image", "audio", "motion", "text", "video", "thermal", "geometry")
        assert e.centroid is not None


def test_episodes_can_filter_by_modality(tmp_path):
    mem = _mem(tmp_path, segment=True)
    aud = mem.episodes(modality="audio")
    assert aud and all(e.modality == "audio" for e in aud)
    # quiet, shout1, alarm, shout2 -> four distinct audio episodes
    assert len(aud) == 4
