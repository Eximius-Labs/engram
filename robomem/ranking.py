"""Three-signal recall ranking (Phase E, part C).

Pure retrieval ranks by one signal: query relevance (cosine). That is the naive baseline and it
is exactly what a caption-then-embed system also does. The memory moat adds two more signals:

* **recency** -- an exponential time-decay toward "now" (or the query time). A "last / most
  recent" question wants the newest relevant moment, not merely the closest one.
* **salience** -- the per-window novelty score from :mod:`robomem.episodes`. A surprising event
  (a sudden violent shake) should surface over a bland one even at equal relevance.

The three signals are min-max normalized across the candidate set before the weighted sum, so
the tunable weights are meaningful and comparable regardless of each signal's native scale. Pure
numpy; no torch, no fusion_embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RankWeights:
    """Weights for the three ranking signals. Only their ratio matters (the sum is normalized)."""

    relevance: float = 1.0
    recency: float = 0.0
    salience: float = 0.0

    @classmethod
    def naive(cls) -> "RankWeights":
        """Pure cosine relevance -- the copyable baseline."""
        return cls(relevance=1.0, recency=0.0, salience=0.0)

    @classmethod
    def three_signal(cls, relevance: float = 1.0, recency: float = 1.0,
                     salience: float = 0.5) -> "RankWeights":
        """The default moat ranking: relevance + recency + a lighter salience nudge."""
        return cls(relevance=relevance, recency=recency, salience=salience)

    def as_dict(self) -> dict:
        return {"relevance": self.relevance, "recency": self.recency, "salience": self.salience}


def recency_score(t: float, now: Optional[float], halflife: float) -> float:
    """Exponential recency in ``[0, 1]``: 1.0 at ``t == now``, halving every ``halflife`` seconds.

    ``now`` None (no reference time) yields 0.0 -- recency contributes nothing. Future events
    (``t > now``) are clamped to the present (score 1.0).
    """
    if now is None:
        return 0.0
    dt = float(now) - float(t)
    if dt <= 0:
        return 1.0
    if halflife <= 0:
        return 0.0
    return float(0.5 ** (dt / float(halflife)))


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        # No spread -> the signal cannot discriminate; treat every candidate as neutral-present.
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def combine(relevance, recency, salience, weights: RankWeights,
            normalize: bool = True) -> np.ndarray:
    """Weighted combination of the three (optionally normalized) signal arrays -> a score array."""
    r = np.asarray(relevance, dtype=np.float64).reshape(-1)
    c = np.asarray(recency, dtype=np.float64).reshape(-1)
    s = np.asarray(salience, dtype=np.float64).reshape(-1)
    if normalize:
        r, c, s = _minmax(r), _minmax(c), _minmax(s)
    total = weights.relevance + weights.recency + weights.salience
    combo = weights.relevance * r + weights.recency * c + weights.salience * s
    return combo / total if total > 0 else combo


def normalized_relevance(relevance) -> np.ndarray:
    """Min-max the relevance scores to ``[0, 1]`` -- the relevance gate the planner keys on."""
    return _minmax(np.asarray(relevance, dtype=np.float64).reshape(-1))
