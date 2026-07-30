"""The Tier-0 ``events`` table schema and the ``Moment`` recall result.

One row per embedded window of a recorded session. The vector column is a full-dim,
L2-normalized embedding produced by the unified embedder (see ``robomem.embedder``); every
modality shares the same width so a text query and an image window are directly comparable.

The schema already carries two Phase-E (episodic / temporal-reasoning) hooks, ``episode_id``
and ``salience``. They are declared nullable now and left unpopulated by the MVP ingest so a
later phase can group events into episodes and score their importance without a migration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pyarrow as pa

# Full interop width of the unified space (Qwen3-VL-Embedding-2B readout).
VECTOR_DIM = 2048


def events_schema(dim: int = VECTOR_DIM) -> pa.Schema:
    """Arrow schema for the Tier-0 ``events`` table.

    Kept explicit (rather than inferred from the first row) so nullable columns and the fixed
    vector width are stable across ingests into the same table.
    """
    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("t_start", pa.float64(), nullable=False),
            pa.field("t_end", pa.float64(), nullable=False),
            pa.field("modality", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=True),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("thumb", pa.string(), nullable=True),
            pa.field("meta", pa.string(), nullable=True),  # JSON blob of extra fields
            # --- Phase-E hooks: declared now, populated later. ---
            pa.field("episode_id", pa.string(), nullable=True),
            pa.field("salience", pa.float32(), nullable=True),
        ]
    )


@dataclass
class Moment:
    """A ranked recall hit: one event row, or a merged segment of adjacent same-source rows."""

    event_id: str
    score: float
    t_start: float
    t_end: float
    modality: str
    source: Optional[str] = None
    meta: dict = field(default_factory=dict)
    thumb: Optional[str] = None
    # Populated when temporally-adjacent same-source hits are merged into a segment.
    member_ids: list = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict, score: float) -> "Moment":
        meta = row.get("meta")
        if isinstance(meta, str) and meta:
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {"_raw": meta}
        elif not isinstance(meta, dict):
            meta = {}
        return cls(
            event_id=row["event_id"],
            score=float(score),
            t_start=float(row["t_start"]),
            t_end=float(row["t_end"]),
            modality=row["modality"],
            source=row.get("source"),
            meta=meta,
            thumb=row.get("thumb"),
            member_ids=[row["event_id"]],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "score": round(self.score, 6),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "modality": self.modality,
            "source": self.source,
            "meta": self.meta,
            "member_ids": self.member_ids,
        }
