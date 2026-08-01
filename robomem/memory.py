"""``RobotMemory`` — open a session store, index a session, answer recall queries.

MVP scope is Phase C (search). The pipeline:

    manifest -> group by modality -> embed (injected embedder) -> LanceDB rows
    recall(text) -> embed query -> SQL prefilter (time / modality) -> exact cosine scan
                 -> optional per-modality centering (cross-modal gap correction)
                 -> merge temporally-adjacent same-source hits -> rank

Query-by-example (:meth:`recall_like`) is the cross-modal, non-visual capability that beats
caption-then-embed systems: hand it any stored vector and ask for the nearest windows of some
other modality, no text round-trip.

Deferred to the temporal / episodic phase (Phase E): populating ``episode_id`` / ``salience``,
episode segmentation, and any "what happened before X" temporal reasoning. The schema already
reserves the columns; nothing here writes them.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from . import temporal
from .embedder import Embedder, l2_normalize, to_numpy
from .episodes import DEFAULT_EPISODE_THRESHOLD, Episode, assign_episodes
from .ingest import build_rows
from .ranking import RankWeights
from .schema import VECTOR_DIM, Moment
from .store import LanceStore


class RobotMemory:
    def __init__(self, store: LanceStore, embedder: Embedder, dim: int = VECTOR_DIM):
        self.store = store
        self.embedder = embedder
        self.dim = dim
        self._centroids: Optional[dict] = None  # per-modality mean vectors, cached
        # Phase-E: change-point threshold + optional temporal-gap break for episode segmentation
        # (used consistently by index and by the temporal planner).
        self.episode_threshold: float = DEFAULT_EPISODE_THRESHOLD
        self.episode_max_gap: Optional[float] = None

    # ------------------------------------------------------------------- open
    @classmethod
    def open(cls, db_path: str, embedder: Embedder, dim: int = VECTOR_DIM,
             create: bool = True) -> "RobotMemory":
        store = LanceStore.open(db_path, dim=dim, create=create)
        return cls(store, embedder, dim)

    # ------------------------------------------------------------------ index
    def index(self, manifest, dedup_tau: Optional[float] = None, segment: bool = False,
              seg_threshold: Optional[float] = None, seg_method: str = "drift",
              seg_max_gap: Optional[float] = None):
        """Ingest a session (manifest path / JSONL / in-memory list) into the table.

        With ``segment=True`` (Phase E) the kept rows are episode-segmented before they land, so
        the ``episode_id`` and ``salience`` schema hooks are populated in place -- no second write
        pass, no LanceDB update. Left off by default so the MVP ingest still writes the hooks as
        NULL. ``seg_threshold`` overrides :attr:`episode_threshold` for this ingest.
        """
        rows, stats = build_rows(self.embedder, manifest, dedup_tau=dedup_tau)
        if segment and rows:
            thr = float(seg_threshold) if seg_threshold is not None else self.episode_threshold
            self.episode_threshold = thr
            if seg_max_gap is not None:
                self.episode_max_gap = float(seg_max_gap)
            ep_by_event, sal_by_event, _ = assign_episodes(
                rows, threshold=thr, method=seg_method, max_gap=self.episode_max_gap)
            for r in rows:
                r["episode_id"] = ep_by_event.get(r["event_id"])
                sal = sal_by_event.get(r["event_id"])
                r["salience"] = None if sal is None else float(sal)
        self.store.add(rows)
        self._centroids = None  # invalidate: new modality means
        stats.skipped = stats.embedded - stats.kept
        return stats

    # ---------------------------------------------------------------- episodes
    def episodes(self, modality=None, seg_threshold: Optional[float] = None,
                 seg_method: str = "drift") -> list[Episode]:
        """Tier-1 episode records built from the stored rows (deterministic, read-only).

        Segments each ``(modality, source)`` stream by embedding change-point and returns
        :class:`~robomem.episodes.Episode` records (time-ordered), each with its member ids,
        centroid, dominant modality, representative label, and peak salience.
        """
        where = self._where(modality=modality)
        rows = self.store.scan(where=where)
        if not rows:
            return []
        thr = float(seg_threshold) if seg_threshold is not None else self.episode_threshold
        _, _, episodes = assign_episodes(rows, threshold=thr, method=seg_method,
                                         max_gap=self.episode_max_gap)
        return episodes

    # -------------------------------------------------------------- centroids
    def _modality_centroids(self) -> dict:
        """Per-modality mean vector over the whole table (for center()-style correction)."""
        if self._centroids is None:
            rows = self.store.scan()
            acc: dict[str, list] = {}
            for r in rows:
                acc.setdefault(r["modality"], []).append(r["vector"])
            self._centroids = {m: np.mean(np.stack(vs), axis=0) for m, vs in acc.items()}
        return self._centroids

    # ------------------------------------------------------------- where SQL
    @staticmethod
    def _where(modality=None, after=None, before=None,
               extra: Optional[str] = None) -> Optional[str]:
        clauses = []
        if modality is not None:
            if isinstance(modality, str):
                clauses.append(f"modality = '{modality}'")
            else:
                opts = ", ".join(f"'{m}'" for m in modality)
                clauses.append(f"modality IN ({opts})")
        if after is not None:
            clauses.append(f"t_end >= {float(after)}")
        if before is not None:
            clauses.append(f"t_start <= {float(before)}")
        if extra:
            clauses.append(extra)
        return " AND ".join(clauses) if clauses else None

    # -------------------------------------------------------------- scoring
    def _score(self, qvec: np.ndarray, q_modality: Optional[str],
               cands: list[dict], center: bool) -> np.ndarray:
        G = np.stack([c["vector"] for c in cands])
        if not center:
            return l2_normalize(G) @ l2_normalize(qvec)
        # Per-modality mean-centering (the recommended cross-modal correction). Subtract each
        # side's modality centroid, renormalize, then take cosine — this removes the modality
        # gap the way UnifiedEmbedder.center() does, estimated from the table.
        cents = self._modality_centroids()
        Gc = G.copy()
        for i, c in enumerate(cands):
            mu = cents.get(c["modality"])
            if mu is not None:
                Gc[i] = G[i] - mu
        Gc = l2_normalize(Gc)
        q = qvec.copy()
        qmu = cents.get(q_modality) if q_modality is not None else None
        if qmu is not None:
            q = q - qmu
        q = l2_normalize(q)
        return Gc @ q

    # --------------------------------------------------------------- retrieval
    def recall(self, query: str, k: int = 10, modality=None,
               after: Optional[float] = None, before: Optional[float] = None,
               center: bool = True, merge: bool = True,
               merge_gap: float = 1.0) -> list[Moment]:
        """Natural-language recall. Returns ranked :class:`Moment` hits (segments if merged)."""
        qvec = to_numpy(self.embedder.embed_text(query))
        return self._recall_with_vector(qvec, "text", k=k, modality=modality, after=after,
                                        before=before, center=center, merge=merge,
                                        merge_gap=merge_gap)

    def recall_like(self, vector, modality_in: Optional[str] = None,
                    return_modality=None, k: int = 10,
                    after: Optional[float] = None, before: Optional[float] = None,
                    center: bool = True, merge: bool = True,
                    merge_gap: float = 1.0) -> list[Moment]:
        """Query-by-example: nearest windows to a given vector.

        ``modality_in`` names the query vector's modality (used only for centering).
        ``return_modality`` restricts the gallery to one or more modalities.
        """
        qvec = to_numpy(vector)
        return self._recall_with_vector(qvec, modality_in, k=k, modality=return_modality,
                                        after=after, before=before, center=center, merge=merge,
                                        merge_gap=merge_gap)

    def _recall_with_vector(self, qvec, q_modality, *, k, modality, after, before,
                            center, merge, merge_gap) -> list[Moment]:
        where = self._where(modality=modality, after=after, before=before)
        cands = self.store.scan(where=where)
        if not cands:
            return []
        scores = self._score(qvec, q_modality, cands, center)
        moments = [Moment.from_row(r, s) for r, s in zip(cands, scores)]
        if merge:
            moments = _merge_segments(moments, merge_gap)
        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:k]

    # --------------------------------------------------------------------- get
    def show(self, event_id: str) -> Optional[dict]:
        row = self.store.get(event_id)
        if row is None:
            return None
        return Moment.from_row(row, score=1.0).as_dict()

    def count(self) -> int:
        return self.store.count()

    # ----------------------------------------------------- temporal reasoning
    # Deterministic chronological operators (Phase E). Thin wrappers over robomem.temporal,
    # which is duck-typed on ``self`` so there is no import cycle.
    def last(self, query: str, modality=None, **kw) -> Optional[Moment]:
        """Most RECENT episode relevant to ``query``. See :func:`robomem.temporal.last`."""
        return temporal.last(self, query, modality=modality, **kw)

    # ``count()`` -> table row count (Phase C, unchanged); ``count("query", modality=...)`` ->
    # number of DISTINCT relevant episodes (Phase E). Dispatched on whether a query is given.
    def count(self, query=None, modality=None, **kw):  # type: ignore[override]
        """Table row count with no args; relevant-episode count when given a query string."""
        if query is None:
            return self.store.count()
        return temporal.count(self, query, modality=modality, **kw)

    def before(self, anchor: str, target: Optional[str] = None, modality=None,
               **kw) -> Optional[Moment]:
        """Nearest relevant ``modality`` episode BEFORE the anchor. :func:`robomem.temporal.before`."""
        return temporal.before(self, anchor, target=target, modality=modality, **kw)

    def after(self, anchor: str, target: Optional[str] = None, modality=None,
              **kw) -> Optional[Moment]:
        """Nearest relevant ``modality`` episode AFTER the anchor. :func:`robomem.temporal.after`."""
        return temporal.after(self, anchor, target=target, modality=modality, **kw)

    def timeline(self, query: Optional[str] = None, window=None, modality=None, **kw):
        """Relevant episodes over a time range, ordered. :func:`robomem.temporal.timeline`."""
        return temporal.timeline(self, query, window=window, modality=modality, **kw)

    def rerank(self, moments, now: Optional[float] = None,
               weights: Optional[RankWeights] = None, halflife: float = 10.0):
        """Re-order Moments by the three-signal score. :func:`robomem.temporal.rerank`."""
        return temporal.rerank(moments, now, weights or RankWeights.three_signal(), halflife)


def _merge_segments(moments: list[Moment], gap: float) -> list[Moment]:
    """Fold temporally-adjacent same-(source, modality) hits into one segment (max score).

    Two hits merge when they share a source and modality and their time ranges are within
    ``gap`` seconds (or overlap). The merged moment spans the union of times, scores at the
    best member, and lists every member id.
    """
    ordered = sorted(moments, key=lambda m: (str(m.source), m.modality, m.t_start))
    out: list[Moment] = []
    for m in ordered:
        if out:
            last = out[-1]
            adjacent = (last.source == m.source and last.modality == m.modality
                        and m.t_start - last.t_end <= gap)
            if adjacent:
                last.t_start = min(last.t_start, m.t_start)
                last.t_end = max(last.t_end, m.t_end)
                last.member_ids += m.member_ids
                if m.score > last.score:  # segment inherits its best member's identity
                    last.score = m.score
                    last.event_id = m.event_id
                    last.meta = m.meta
                    last.thumb = m.thumb
                continue
        out.append(m)
    return out
