"""Deterministic temporal query planner (Phase E, part D).

Pure retrieval answers "what looks like X". A robot's memory has to answer *chronological*
questions retrieval alone cannot: **when** did X last happen, **how many times**, what happened
**before / after** it, and **show me the timeline**. These are the queries buyers actually ask,
and they are the moat over caption-then-embed.

Every operator here is deterministic -- no LLM decomposition in v0. Each is a small program over
two primitives the memory layer already has: a LanceDB time/modality prefilter, and cosine
relevance from the injected embedder. Episodes (:mod:`robomem.episodes`) group contiguous windows
so "how many times" counts distinct occurrences, not raw windows.

Operators (wired onto :class:`robomem.memory.RobotMemory` as ``mem.last`` / ``mem.count`` /
``mem.before`` / ``mem.after`` / ``mem.timeline``):

* ``last``   -- the most recent episode relevant to a query.
* ``count``  -- how many distinct relevant episodes.
* ``before`` / ``after`` -- find an anchor event's time, then the nearest relevant target
  episode on the requested side of it ("what did it see right before the alarm").
* ``timeline`` -- relevant episodes over a time range, in order.

This module imports only numpy + robomem's own pure helpers. ``mem`` is duck-typed, so there is
no import cycle with :mod:`robomem.memory`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .embedder import to_numpy
from .episodes import DEFAULT_EPISODE_THRESHOLD, assign_episodes
from .ranking import RankWeights, combine, normalized_relevance, recency_score
from .schema import Moment

# Default fraction of the (min-max normalized) relevance range a candidate must clear to count as
# "relevant" to a query. 0.5 keeps the upper half; strongly-matching episodes clear it, weak ones
# do not. Tunable per call.
DEFAULT_REL_KEEP = 0.5


class _RelEpisode:
    """An episode paired with its query relevance + the best-matching member row (for a Moment)."""

    __slots__ = ("ep", "relevance", "best_id", "best_row")

    def __init__(self, ep, relevance, best_id, best_row):
        self.ep = ep
        self.relevance = relevance
        self.best_id = best_id
        self.best_row = best_row


def _seg_threshold(mem, threshold) -> float:
    if threshold is not None:
        return float(threshold)
    return float(getattr(mem, "episode_threshold", DEFAULT_EPISODE_THRESHOLD))


def _seg_gap(mem):
    return getattr(mem, "episode_max_gap", None)


def _relevant_episodes(mem, qvec, q_modality, modality, center, threshold) -> list:
    """Scan the modality-filtered stream, score relevance, and fold into relevance-ranked episodes.

    Episodes are segmented over the whole modality-filtered set (all times) so a stream is never
    cut mid-episode; time windows are applied by the callers afterward. Each episode's relevance
    is the relevance of its best-matching member.
    """
    where = mem._where(modality=modality)
    rows = mem.store.scan(where=where)
    if not rows:
        return []
    scores = mem._score(to_numpy(qvec), q_modality, rows, center)
    rel = {r["event_id"]: float(s) for r, s in zip(rows, scores)}
    rowmap = {r["event_id"]: r for r in rows}
    _, _, episodes = assign_episodes(rows, threshold=_seg_threshold(mem, threshold), max_gap=_seg_gap(mem))
    out: list[_RelEpisode] = []
    for ep in episodes:
        best_id = max(ep.member_ids, key=lambda e: rel[e])
        out.append(_RelEpisode(ep, rel[best_id], best_id, rowmap[best_id]))
    return out


def _gate(rel_eps, rel_keep, min_relevance) -> list:
    """Keep episodes whose (normalized) relevance clears the gate."""
    if not rel_eps:
        return []
    norm = normalized_relevance([re.relevance for re in rel_eps])
    kept = []
    for re, nrel in zip(rel_eps, norm):
        if nrel < rel_keep:
            continue
        if min_relevance is not None and re.relevance < min_relevance:
            continue
        kept.append(re)
    return kept


def _episode_moment(re: _RelEpisode) -> Moment:
    """Build a :class:`Moment` spanning the episode, scored at its best member, carrying members."""
    m = Moment.from_row(re.best_row, re.relevance)
    m.t_start = float(re.ep.t_start)
    m.t_end = float(re.ep.t_end)
    m.member_ids = list(re.ep.member_ids)
    meta = dict(m.meta) if isinstance(m.meta, dict) else {}
    meta["episode_id"] = re.ep.episode_id
    meta["episode_salience"] = round(float(re.ep.salience), 6)
    if re.ep.label is not None:
        meta["episode_label"] = re.ep.label
    m.meta = meta
    return m


# --------------------------------------------------------------------------- operators
def last(mem, query: str, modality=None, *, center: bool = True,
         threshold: Optional[float] = None, rel_keep: float = DEFAULT_REL_KEEP,
         min_relevance: Optional[float] = None) -> Optional[Moment]:
    """The most RECENT episode relevant to ``query`` (optionally restricted to ``modality``)."""
    qvec = mem.embedder.embed_text(query)
    rel_eps = _relevant_episodes(mem, qvec, "text", modality, center, threshold)
    kept = _gate(rel_eps, rel_keep, min_relevance)
    if not kept:
        return None
    best = max(kept, key=lambda re: (re.ep.t_start, re.relevance))
    return _episode_moment(best)


def count(mem, query: str, modality=None, *, center: bool = True,
          threshold: Optional[float] = None, rel_keep: float = DEFAULT_REL_KEEP,
          min_relevance: Optional[float] = None) -> int:
    """How many DISTINCT relevant episodes match ``query`` (contiguous windows count once)."""
    qvec = mem.embedder.embed_text(query)
    rel_eps = _relevant_episodes(mem, qvec, "text", modality, center, threshold)
    return len(_gate(rel_eps, rel_keep, min_relevance))


def _anchor_time(mem, anchor, anchor_modality, center) -> Optional[float]:
    """Onset time of the single most relevant event to ``anchor`` (in ``anchor_modality``)."""
    hits = mem.recall(anchor, modality=anchor_modality, k=1, center=center, merge=False)
    return float(hits[0].t_start) if hits else None


def before(mem, anchor: str, target: Optional[str] = None, modality=None, *,
           anchor_modality=None, center: bool = True, threshold: Optional[float] = None,
           rel_keep: float = DEFAULT_REL_KEEP,
           min_relevance: Optional[float] = None) -> Optional[Moment]:
    """The nearest relevant ``modality`` episode occurring BEFORE the ``anchor`` event.

    Finds the anchor event's time (top relevance for ``anchor`` within ``anchor_modality``), then
    returns the ``modality`` episode with the largest ``t_start`` still before it. When ``target``
    is given the candidate episodes are also relevance-gated against it; otherwise the nearest
    preceding episode of ``modality`` (whatever its content) is returned.
    """
    return _side(mem, anchor, target, modality, "before", anchor_modality, center,
                 threshold, rel_keep, min_relevance)


def after(mem, anchor: str, target: Optional[str] = None, modality=None, *,
          anchor_modality=None, center: bool = True, threshold: Optional[float] = None,
          rel_keep: float = DEFAULT_REL_KEEP,
          min_relevance: Optional[float] = None) -> Optional[Moment]:
    """The nearest relevant ``modality`` episode occurring AFTER the ``anchor`` event."""
    return _side(mem, anchor, target, modality, "after", anchor_modality, center,
                 threshold, rel_keep, min_relevance)


def _side(mem, anchor, target, modality, side, anchor_modality, center, threshold,
          rel_keep, min_relevance) -> Optional[Moment]:
    t_anchor = _anchor_time(mem, anchor, anchor_modality, center)
    if t_anchor is None:
        return None
    if target is not None:
        qvec = mem.embedder.embed_text(target)
        rel_eps = _relevant_episodes(mem, qvec, "text", modality, center, threshold)
        cands = _gate(rel_eps, rel_keep, min_relevance)
    else:
        # No target content filter: every episode of ``modality`` is a candidate.
        where = mem._where(modality=modality)
        rows = mem.store.scan(where=where)
        if not rows:
            return None
        _, _, episodes = assign_episodes(rows, threshold=_seg_threshold(mem, threshold), max_gap=_seg_gap(mem))
        rowmap = {r["event_id"]: r for r in rows}
        cands = [_RelEpisode(ep, 1.0, ep.member_ids[0], rowmap[ep.member_ids[0]])
                 for ep in episodes]
    if not cands:
        return None
    if side == "before":
        pool = [re for re in cands if re.ep.t_start < t_anchor - 1e-9]
        if not pool:
            return None
        pick = max(pool, key=lambda re: re.ep.t_start)   # nearest below the anchor
    else:
        pool = [re for re in cands if re.ep.t_start > t_anchor + 1e-9]
        if not pool:
            return None
        pick = min(pool, key=lambda re: re.ep.t_start)   # nearest above the anchor
    m = _episode_moment(pick)
    m.meta = {**m.meta, "anchor_query": anchor, "anchor_time": round(t_anchor, 6), "side": side}
    return m


def timeline(mem, query: Optional[str] = None, *, window=None, modality=None,
             center: bool = True, threshold: Optional[float] = None,
             rel_keep: float = DEFAULT_REL_KEEP,
             min_relevance: Optional[float] = None) -> list:
    """Relevant episodes over a time range, ordered by start time.

    ``window`` is an optional ``(t0, t1)`` bound applied to episode start times. With no ``query``
    every episode is returned (an ordered index of the session); with a ``query`` the episodes are
    relevance-gated first.
    """
    if query is not None:
        qvec = mem.embedder.embed_text(query)
        rel_eps = _relevant_episodes(mem, qvec, "text", modality, center, threshold)
        cands = _gate(rel_eps, rel_keep, min_relevance)
    else:
        where = mem._where(modality=modality)
        rows = mem.store.scan(where=where)
        if not rows:
            return []
        _, _, episodes = assign_episodes(rows, threshold=_seg_threshold(mem, threshold), max_gap=_seg_gap(mem))
        rowmap = {r["event_id"]: r for r in rows}
        cands = [_RelEpisode(ep, 1.0, ep.member_ids[0], rowmap[ep.member_ids[0]])
                 for ep in episodes]
    if window is not None:
        t0, t1 = window
        cands = [re for re in cands
                 if (t0 is None or re.ep.t_start >= t0) and (t1 is None or re.ep.t_start <= t1)]
    cands.sort(key=lambda re: re.ep.t_start)
    return [_episode_moment(re) for re in cands]


def rerank(moments, now: Optional[float], weights: RankWeights, halflife: float) -> list:
    """Re-order ``Moment`` hits by the three-signal score (relevance + recency + salience).

    Relevance is each moment's cosine ``score``; recency is the time decay of its start toward
    ``now``; salience is ``meta.episode_salience`` (0.0 when absent). Returns a new sorted list.
    Pure cosine ordering is recovered with :meth:`RankWeights.naive`.
    """
    moments = list(moments)
    if not moments:
        return moments
    rel = [m.score for m in moments]
    rec = [recency_score(m.t_start, now, halflife) for m in moments]
    sal = [float(m.meta.get("episode_salience", 0.0)) if isinstance(m.meta, dict) else 0.0
           for m in moments]
    scores = combine(rel, rec, sal, weights)
    order = sorted(range(len(moments)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        m = moments[i]
        if isinstance(m.meta, dict):
            m.meta = {**m.meta, "rank_score": round(float(scores[i]), 6)}
        out.append(m)
    return out
