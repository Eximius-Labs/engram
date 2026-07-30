"""Session ingestion: manifest -> group by modality -> embed -> rows.

A *session* is a list of timestamped media events. The MVP does not require real robot logs;
it reads a simple manifest (a JSONL file, a JSON array, or an in-memory list) of events:

    {"t": 12.5, "modality": "image", "path_or_data": "cam0/frame_00012.png",
     "source": "cam0", "duration": 0.0, "meta": {...}}

Required per event: ``t`` (or ``t_start``) and ``modality`` and ``path_or_data``. Everything
else is optional. Events are grouped BY MODALITY before embedding because the unified embedder
is gate-based: per-modality calls keep the adapter gates safe (text/image/video with all gates
closed, audio with only the audio gate, thermal with only the thermal gate).

``dedup_tau`` is the edge-storage lever: within a stream (same modality + source, in time
order) a window whose cosine similarity to the previous *kept* window exceeds ``tau`` is
dropped, so a static camera staring at a wall does not fill the table with duplicates.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .embedder import Embedder, l2_normalize, to_numpy

_MODALITIES = ("text", "image", "video", "audio", "thermal", "motion", "geometry")


@dataclass
class IngestStats:
    embedded: int = 0
    kept: int = 0
    deduped: int = 0
    skipped: int = 0
    per_modality: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "embedded": self.embedded, "kept": self.kept,
            "deduped": self.deduped, "skipped": self.skipped,
            "per_modality": self.per_modality,
        }


def read_manifest(manifest) -> list[dict]:
    """Load a session from a JSONL / JSON-array file path, or pass through an in-memory list."""
    if isinstance(manifest, (list, tuple)):
        return list(manifest)
    path = Path(manifest)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _norm_event(ev: dict, idx: int) -> dict:
    m = ev.get("modality")
    if m not in _MODALITIES:
        raise ValueError(f"event {idx}: unknown modality {m!r}; known: {_MODALITIES}")
    if "path_or_data" not in ev:
        raise ValueError(f"event {idx}: missing 'path_or_data'")
    t_start = ev.get("t_start", ev.get("t"))
    if t_start is None:
        raise ValueError(f"event {idx}: missing 't' / 't_start'")
    duration = float(ev.get("duration", 0.0))
    t_end = ev.get("t_end", float(t_start) + duration)
    source = ev.get("source")
    if source is None and isinstance(ev["path_or_data"], (str, os.PathLike)):
        # A sensible default stream key: the immediate parent dir of the media path.
        source = Path(str(ev["path_or_data"])).parent.name or None
    return {
        "event_id": str(ev.get("id") or ev.get("event_id") or f"ev_{idx}_{uuid.uuid4().hex[:8]}"),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "modality": m,
        "source": source,
        "payload": ev["path_or_data"],
        "thumb": ev.get("thumb"),
        "meta": ev.get("meta") or {},
    }


def _embed_one(embedder: Embedder, modality: str, payload) -> np.ndarray:
    """Dispatch a single event to the right embed_* method and return a numpy vector.

    Paths flow straight through: UnifiedEmbedder.embed_image / embed_audio accept a path and do
    their own loading, and the fake hashes the string, so no loader lives here for the common
    case. Audio / motion arrays may be passed as ``{"data": [...], "sr": N}``.
    """
    if modality == "text":
        v = embedder.embed_text(str(payload))
    elif modality == "image":
        v = embedder.embed_image(payload)
    elif modality == "video":
        v = embedder.embed_video(payload)
    elif modality == "thermal":
        v = embedder.embed_thermal(payload)
    elif modality == "geometry":
        v = embedder.embed_geometry(payload)
    elif modality == "audio":
        if isinstance(payload, dict):
            v = embedder.embed_audio(np.asarray(payload["data"], dtype=np.float32),
                                     sr=payload.get("sr"))
        else:
            v = embedder.embed_audio(payload)
    elif modality == "motion":
        data = payload["data"] if isinstance(payload, dict) else payload
        v = embedder.embed_motion(np.asarray(data, dtype=np.float32))
    else:  # pragma: no cover - guarded by _norm_event
        raise ValueError(modality)
    return to_numpy(v)


def build_rows(embedder: Embedder, events, dedup_tau: Optional[float] = None) -> tuple[list[dict], IngestStats]:
    """Embed a session into Tier-0 rows, grouped by modality and (optionally) deduplicated.

    Returns ``(rows, stats)``. ``rows`` are ready for :meth:`robomem.store.LanceStore.add`.
    """
    raw = read_manifest(events) if not isinstance(events, (list, tuple)) or (
        events and isinstance(events[0], (str, os.PathLike))
    ) else list(events)
    norm = [_norm_event(ev, i) for i, ev in enumerate(raw)]

    # Group by modality (gate-safe), preserving first-seen modality order.
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for ev in norm:
        groups.setdefault(ev["modality"], []).append(ev)
        if ev["modality"] not in order:
            order.append(ev["modality"])

    stats = IngestStats()
    rows: list[dict] = []
    for modality in order:
        # Within a modality, dedup runs per stream (source) in time order.
        last_kept: dict[Optional[str], np.ndarray] = {}
        for ev in sorted(groups[modality], key=lambda e: e["t_start"]):
            vec = _embed_one(embedder, modality, ev["payload"])
            stats.embedded += 1
            stats.per_modality.setdefault(modality, {"embedded": 0, "kept": 0, "deduped": 0})
            stats.per_modality[modality]["embedded"] += 1

            if dedup_tau is not None:
                prev = last_kept.get(ev["source"])
                if prev is not None:
                    sim = float(np.dot(l2_normalize(vec), l2_normalize(prev)))
                    if sim > dedup_tau:
                        stats.deduped += 1
                        stats.per_modality[modality]["deduped"] += 1
                        continue
                last_kept[ev["source"]] = vec

            rows.append({
                "event_id": ev["event_id"],
                "t_start": ev["t_start"],
                "t_end": ev["t_end"],
                "modality": modality,
                "source": ev["source"],
                "vector": vec,
                "thumb": ev["thumb"],
                "meta": json.dumps(ev["meta"]) if ev["meta"] else None,
                "episode_id": None,   # Phase-E hook
                "salience": None,     # Phase-E hook
            })
            stats.kept += 1
            stats.per_modality[modality]["kept"] += 1

    return rows, stats
