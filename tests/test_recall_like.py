"""(c) Query-by-example: a cross-modal recall_like works (image vector -> matching audio)."""


def test_image_vector_recalls_matching_audio(mem, embedder):
    dog_img = embedder.embed_image("cam0/dog_01.png")
    hits = mem.recall_like(dog_img, modality_in="image", return_modality="audio", k=5)
    assert hits, "no cross-modal hits"
    assert all(h.modality == "audio" for h in hits)
    assert hits[0].event_id == "aud_dog"


def test_cross_modal_beats_the_distractor(mem, embedder):
    cat_img = embedder.embed_image("cam0/cat_01.png")
    hits = mem.recall_like(cat_img, modality_in="image", return_modality="audio", k=5)
    assert hits[0].event_id == "aud_cat"


def test_recall_like_can_span_all_modalities(mem, embedder):
    dog_img = embedder.embed_image("cam0/dog_01.png")
    hits = mem.recall_like(dog_img, modality_in="image", k=10)
    # its own image plus at least one other modality of the same concept
    assert len(hits) >= 2
    assert len({h.modality for h in hits}) >= 2
