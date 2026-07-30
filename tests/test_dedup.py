"""(e) Embedding dedup drops a near-duplicate window."""


def test_dedup_drops_near_duplicate(make_mem):
    mem, stats = make_mem(dedup_tau=0.98)
    # img_dog2 is the near-duplicate of img_dog1 (same concept, same camera) -> dropped.
    assert stats.deduped == 1
    assert stats.kept == stats.embedded - 1
    assert mem.store.get("img_dog2") is None
    assert mem.store.get("img_dog1") is not None
    assert mem.store.get("img_cat") is not None  # a genuinely different window survives


def test_no_dedup_keeps_everything(make_mem):
    mem, stats = make_mem(dedup_tau=None)
    assert stats.deduped == 0
    assert mem.store.get("img_dog2") is not None


def test_dedup_is_per_stream(make_mem):
    # dedup keys on (modality, source); the dog audio must not be dropped against the dog image.
    mem, stats = make_mem(dedup_tau=0.98)
    assert mem.store.get("aud_dog") is not None
    assert stats.per_modality["audio"]["deduped"] == 0
