"""Run development or integrated evaluations."""

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from .contracts import CONFIGURATIONS
from .evaluation import EvaluationHarness, score_run
from .fixtures import FixtureSystem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="Import correctness and citation-entailment judgments")
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--ratings", required=True, type=Path)
    score.add_argument("--citations", required=True, type=Path)
    score.add_argument("--out", required=True, type=Path)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--rows", default="all", help="all or comma-separated configuration names")
    evaluate.add_argument("--snapshot", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)
    evaluate.add_argument("--backend", choices=("fixture", "integrated"), required=True)
    evaluate.add_argument("--adapter", help="Integration module:factory; factory receives items.jsonl path only")
    evaluate.add_argument("--adapter-revision", help="Identifier for the adapter revision and fixed run settings")
    evaluate.add_argument("--resume", action="store_true", help="Continue pending and unmeasured cases")
    evaluate.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.command == "score":
        rows = score_run(args.run, args.ratings, args.citations, args.out)
        print(json.dumps({"rows": len(rows), "output": str(args.out)}))
        return
    if args.backend == "fixture":
        if args.adapter or args.adapter_revision:
            parser.error("--adapter and --adapter-revision are only used with --backend integrated")
        adapter = FixtureSystem(args.snapshot / "items.jsonl")
        run_kind = "fixture_smoke"
    else:
        if not args.adapter or ":" not in args.adapter:
            parser.error("Integrated execution requires --adapter module:factory")
        if not args.adapter_revision or not args.adapter_revision.strip():
            parser.error("Integrated execution requires --adapter-revision")
        module, factory = args.adapter.split(":", 1)
        adapter = getattr(importlib.import_module(module), factory)(args.snapshot / "items.jsonl")
        run_kind = "integrated"
    configs = CONFIGURATIONS
    if args.rows != "all":
        wanted = args.rows.split(",")
        if len(set(wanted)) != len(wanted) or set(wanted) - {c.name for c in configs}:
            parser.error("Unknown or repeated configuration name")
        configs = tuple(c for c in configs if c.name in wanted)
    harness = EvaluationHarness(adapter, run_kind, args.seed, adapter_revision=args.adapter_revision)
    asyncio.run(harness.run(args.snapshot, args.out, configs, resume=args.resume))
    metadata = json.loads((args.out / "run.json").read_text(encoding="utf-8"))
    fields = ("run_kind", "run_status", "measured", "unmeasured", "pending")
    print(json.dumps({**{key: metadata[key] for key in fields}, "output": str(args.out)}))
    return 0 if metadata["run_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
