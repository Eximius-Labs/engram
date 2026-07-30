"""robomem command line: index a session, query it, inspect an event.

    robomem index <manifest> --db <path> [--dedup-tau 0.98] [--fake]
    robomem query "<text>" --db <path> [--modality image] [-k 10] [--fake]
    robomem show <event_id> --db <path> [--fake]

By default the CLI loads the real embedder via
``fusion_embedding.unified.UnifiedEmbedder.from_pretrained(--model, device=--device)``.
Pass ``--fake`` to use the deterministic CPU stand-in (handy for demos and for querying a DB
that was indexed with the fake).
"""

from __future__ import annotations

import argparse
import json
import sys

from .memory import RobotMemory


def _load_embedder(args):
    if getattr(args, "fake", False):
        from .fakes import FakeEmbedder
        return FakeEmbedder()
    from fusion_embedding.unified import UnifiedEmbedder
    return UnifiedEmbedder.from_pretrained(args.model, device=args.device)


def _add_common(p):
    p.add_argument("--db", required=True, help="LanceDB directory for this session store")
    p.add_argument("--fake", action="store_true", help="use the CPU stand-in embedder")
    p.add_argument("--model", default="EximiusLabs/fusion-embedding-2-2b-preview",
                   help="repo id / path for the real embedder")
    p.add_argument("--device", default="cuda", help="device for the real embedder")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="robomem", description="robot-memory recall layer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="ingest a session manifest into a store")
    p_idx.add_argument("manifest", help="JSONL / JSON-array manifest of timestamped events")
    p_idx.add_argument("--dedup-tau", type=float, default=None,
                       help="drop a window whose cosine to the previous kept one exceeds tau")
    _add_common(p_idx)

    p_q = sub.add_parser("query", help="natural-language recall over a store")
    p_q.add_argument("text", help="the recall query")
    p_q.add_argument("-k", type=int, default=10)
    p_q.add_argument("--modality", nargs="+", default=None, help="restrict to modality/-ies")
    p_q.add_argument("--after", type=float, default=None)
    p_q.add_argument("--before", type=float, default=None)
    p_q.add_argument("--no-center", action="store_true", help="disable cross-modal centering")
    p_q.add_argument("--no-merge", action="store_true", help="do not merge adjacent hits")
    _add_common(p_q)

    p_s = sub.add_parser("show", help="print one event row by id")
    p_s.add_argument("event_id")
    _add_common(p_s)

    args = ap.parse_args(argv)
    mem = RobotMemory.open(args.db, embedder=_load_embedder(args),
                           create=(args.cmd == "index"))

    if args.cmd == "index":
        stats = mem.index(args.manifest, dedup_tau=args.dedup_tau)
        print(json.dumps({"indexed": stats.as_dict(), "table_rows": mem.count()}, indent=2))
        return 0

    if args.cmd == "query":
        modality = args.modality[0] if (args.modality and len(args.modality) == 1) else args.modality
        hits = mem.recall(args.text, k=args.k, modality=modality, after=args.after,
                          before=args.before, center=not args.no_center, merge=not args.no_merge)
        print(json.dumps([h.as_dict() for h in hits], indent=2))
        return 0

    if args.cmd == "show":
        row = mem.show(args.event_id)
        if row is None:
            print(f"no event {args.event_id!r}", file=sys.stderr)
            return 1
        print(json.dumps(row, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
