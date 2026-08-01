"""Phase E (C): three-signal ranking = relevance + recency + salience."""

import numpy as np

from robomem.ranking import (
    RankWeights,
    combine,
    normalized_relevance,
    recency_score,
)


# --------------------------------------------------------------------- recency
def test_recency_is_one_at_now_and_halves_at_halflife():
    assert recency_score(10.0, now=10.0, halflife=5.0) == 1.0
    assert abs(recency_score(5.0, now=10.0, halflife=5.0) - 0.5) < 1e-9
    assert abs(recency_score(0.0, now=10.0, halflife=5.0) - 0.25) < 1e-9


def test_recency_monotonic_and_edge_cases():
    now = 100.0
    scores = [recency_score(t, now=now, halflife=10.0) for t in (10, 30, 60, 90, 100)]
    assert scores == sorted(scores)          # older -> smaller
    assert recency_score(50.0, now=None, halflife=10.0) == 0.0   # no reference time
    assert recency_score(120.0, now=100.0, halflife=10.0) == 1.0  # future clamps to present


# --------------------------------------------------------------------- combine
def test_naive_weights_recover_relevance_order():
    rel = [0.1, 0.9, 0.5]
    rec = [1.0, 0.0, 0.5]
    sal = [0.0, 0.0, 1.0]
    scores = combine(rel, rec, sal, RankWeights.naive())
    # naive == relevance only -> argmax is the most relevant (index 1)
    assert int(np.argmax(scores)) == 1


def test_recency_weight_can_override_a_relevance_tie():
    # two equally relevant candidates, different recency -> the newer one wins
    rel = [0.8, 0.8]
    rec = [0.2, 1.0]
    sal = [0.0, 0.0]
    scores = combine(rel, rec, sal, RankWeights.three_signal())
    assert int(np.argmax(scores)) == 1


def test_salience_breaks_a_relevance_tie():
    rel = [0.8, 0.8]
    rec = [0.5, 0.5]
    sal = [0.1, 0.9]
    scores = combine(rel, rec, sal, RankWeights(relevance=1.0, recency=0.0, salience=1.0))
    assert int(np.argmax(scores)) == 1


def test_weights_normalize_so_only_ratios_matter():
    rel, rec, sal = [0.2, 0.9], [0.9, 0.1], [0.0, 0.0]
    a = combine(rel, rec, sal, RankWeights(relevance=1.0, recency=1.0, salience=0.0))
    b = combine(rel, rec, sal, RankWeights(relevance=2.0, recency=2.0, salience=0.0))
    assert np.allclose(a, b)


def test_normalized_relevance_is_unit_range():
    n = normalized_relevance([0.1, 0.5, 0.9])
    assert n.min() == 0.0 and n.max() == 1.0
    # a degenerate all-equal signal maps to all-ones (no discrimination)
    assert np.allclose(normalized_relevance([0.4, 0.4, 0.4]), 1.0)


def test_rankweights_helpers():
    assert RankWeights.naive().as_dict() == {"relevance": 1.0, "recency": 0.0, "salience": 0.0}
    ts = RankWeights.three_signal()
    assert ts.recency > 0 and ts.relevance > 0
