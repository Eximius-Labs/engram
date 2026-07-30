"""Thin LanceDB wrapper for the Tier-0 ``events`` table.

LanceDB is used as embedded columnar storage plus a SQL prefilter. At session scale the
retrieval is an exact numpy scan over the filtered candidate set (see ``robomem.memory``), so
this layer only needs to create the table, append rows, and return rows matching a SQL
predicate on time / modality. That keeps behavior deterministic and ANN-index-free for the
MVP while leaving the door open to LanceDB's vector index later.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .schema import VECTOR_DIM, events_schema

TABLE = "events"


class LanceStore:
    def __init__(self, db, table, dim: int):
        self.db = db
        self.table = table
        self.dim = dim

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, db_path: str, dim: int = VECTOR_DIM, create: bool = True) -> "LanceStore":
        import lancedb

        db = lancedb.connect(str(db_path))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = set(db.table_names())
        if TABLE in names:
            table = db.open_table(TABLE)
        elif create:
            table = db.create_table(TABLE, schema=events_schema(dim))
        else:
            raise FileNotFoundError(f"no '{TABLE}' table in {db_path!r}; index a session first")
        return cls(db, table, dim)

    # ------------------------------------------------------------------ write
    def add(self, rows: list[dict]) -> int:
        """Append rows. Each row's ``vector`` must be a length-``dim`` sequence of floats."""
        if not rows:
            return 0
        clean = []
        for r in rows:
            v = np.asarray(r["vector"], dtype=np.float32).reshape(-1)
            if v.shape[0] != self.dim:
                raise ValueError(f"vector dim {v.shape[0]} != table dim {self.dim} "
                                 f"(event {r.get('event_id')!r})")
            row = dict(r)
            row["vector"] = v.tolist()
            row.setdefault("source", None)
            row.setdefault("thumb", None)
            row.setdefault("meta", None)
            row.setdefault("episode_id", None)   # Phase-E hook, unpopulated
            row.setdefault("salience", None)     # Phase-E hook, unpopulated
            clean.append(row)
        self.table.add(clean)
        return len(clean)

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.table.count_rows()

    def scan(self, where: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
        """Return rows matching an optional SQL ``where`` predicate (columnar full scan).

        Vectors come back as numpy float32 arrays. LanceDB applies the predicate; if a given
        build rejects a vector-less search we fall back to an in-process filtered read so the
        contract (a filtered list of row dicts) is identical either way.
        """
        rows = self._raw_scan(where, limit)
        for r in rows:
            r["vector"] = np.asarray(r.get("vector"), dtype=np.float32).reshape(-1)
        return rows

    def _raw_scan(self, where: Optional[str], limit: Optional[int]) -> list[dict]:
        try:
            q = self.table.search()
            if where:
                q = q.where(where, prefilter=True)
            if limit is not None:
                q = q.limit(limit)
            else:
                q = q.limit(self.count() or 1)
            return q.to_list()
        except Exception:
            # Version-robust fallback: read the whole table and filter with pyarrow.
            tbl = self.table.to_arrow()
            rows = tbl.to_pylist()
            if where:
                rows = [r for r in rows if _py_predicate(where, r)]
            if limit is not None:
                rows = rows[:limit]
            return rows

    def get(self, event_id: str) -> Optional[dict]:
        safe = event_id.replace("'", "''")
        rows = self.scan(where=f"event_id = '{safe}'", limit=1)
        return rows[0] if rows else None


def _py_predicate(where: str, row: dict) -> bool:
    """Minimal evaluator for the AND-of-comparisons predicates robomem builds (fallback path)."""
    for clause in where.split(" AND "):
        c = clause.strip()
        ok = True
        if " IN (" in c:
            col, rest = c.split(" IN (", 1)
            opts = {s.strip().strip("'") for s in rest.rstrip(")").split(",")}
            ok = str(row.get(col.strip())) in opts
        else:
            for op in (">=", "<=", "=", ">", "<"):
                if op in c:
                    col, val = c.split(op, 1)
                    col, val = col.strip(), val.strip().strip("'")
                    lhs = row.get(col)
                    try:
                        rhs = float(val)
                        lhs = float(lhs)
                    except (TypeError, ValueError):
                        rhs = val
                    ok = {
                        ">=": lhs >= rhs, "<=": lhs <= rhs, "=": lhs == rhs,
                        ">": lhs > rhs, "<": lhs < rhs,
                    }[op]
                    break
        if not ok:
            return False
    return True
