"""(b) A text query recalls the correct planted clip."""


def test_text_query_recalls_planted_image(mem):
    hits = mem.recall("dog", modality="image", k=5)
    assert hits, "no hits returned"
    assert hits[0].event_id.startswith("img_dog")
    assert hits[0].modality == "image"
    # the cat image must rank below the dog image
    ids = [h.event_id for h in hits]
    assert ids.index("img_cat") > 0 if "img_cat" in ids else True


def test_text_query_distinguishes_concepts(mem):
    dog = mem.recall("dog", modality="image", k=5)[0]
    cat = mem.recall("cat", modality="image", k=5)[0]
    assert dog.event_id.startswith("img_dog")
    assert cat.event_id == "img_cat"


def test_recall_without_modality_filter_ranks_dog_first(mem):
    hits = mem.recall("dog", k=10)
    assert hits
    # top hit is some dog-concept event, not a cat one
    assert "cat" not in mem.show(hits[0].event_id)["meta"].get("path", "dog")


def test_scores_are_sorted_descending(mem):
    hits = mem.recall("dog", k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
