"""(d) Time and modality prefilters work."""


def test_modality_filter_restricts_gallery(mem):
    hits = mem.recall("dog", modality="audio", k=10)
    assert hits
    assert all(h.modality == "audio" for h in hits)
    assert hits[0].event_id == "aud_dog"


def test_modality_filter_accepts_a_list(mem):
    hits = mem.recall("dog", modality=["image", "audio"], k=10)
    assert {h.modality for h in hits} <= {"image", "audio"}


def test_before_filter_excludes_late_events(mem):
    hits = mem.recall("dog", before=3.0, k=10)
    assert hits
    for h in hits:
        assert h.t_start <= 3.0
    # the cat events (t >= 5.0) must be gone
    assert all(not h.event_id.endswith("cat") for h in hits)


def test_after_filter_excludes_early_events(mem):
    hits = mem.recall("cat", after=5.0, k=10)
    assert hits
    for h in hits:
        assert h.t_end >= 5.0


def test_window_filter_combines_bounds(mem):
    hits = mem.recall("dog", after=1.0, before=2.5, k=10)
    assert hits
    for h in hits:
        assert h.t_start <= 2.5 and h.t_end >= 1.0
