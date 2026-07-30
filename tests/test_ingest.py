"""(a) Ingest -> index writes the right rows, with the Phase-E columns present but empty."""

from tests.conftest import session_events


def test_index_writes_one_row_per_event(make_mem):
    mem, stats = make_mem()
    events = session_events()
    assert mem.count() == len(events)
    assert stats.embedded == len(events)
    assert stats.kept == len(events)
    assert stats.deduped == 0


def test_every_modality_and_fields_present(mem):
    rows = mem.store.scan()
    modalities = {r["modality"] for r in rows}
    assert modalities == {"image", "audio", "text", "motion", "thermal"}
    for r in rows:
        assert r["vector"].shape == (mem.dim,)
        assert r["t_end"] >= r["t_start"]
        assert r["source"] is not None


def test_phase_e_hooks_declared_but_unpopulated(mem):
    row = mem.store.get("img_dog1")
    assert row is not None
    assert "episode_id" in row and row["episode_id"] is None
    assert "salience" in row and row["salience"] is None


def test_show_returns_event(mem):
    shown = mem.show("aud_dog")
    assert shown["event_id"] == "aud_dog"
    assert shown["modality"] == "audio"
    assert shown["meta"]["path"] == "mic0/dog_bark.wav"
    assert mem.show("does_not_exist") is None
