"""Run the test suite and development evaluation."""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def test_groups(suite):
    groups = Counter()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            groups.update(test_groups(test))
        else:
            groups[type(test).__name__] += 1
    return groups


async def check_evaluation(output):
    from omnistream.contracts import CONFIGURATIONS
    from omnistream.evaluation import EvaluationHarness
    from omnistream.fixtures import FixtureSystem

    snapshot = ROOT / "data/pilot_snapshot"
    rows = await EvaluationHarness(FixtureSystem(snapshot / "items.jsonl"), "fixture_smoke").run(snapshot, output)
    expected = {config.name: 12 for config in CONFIGURATIONS}
    if dict(Counter(row["configuration"] for row in rows)) != expected:
        raise ValueError("Expected twelve questions for each of the six configurations")
    if any(row["status"] != "ok" or row["error"] for row in rows):
        raise ValueError("One or more development runs failed")
    if any(row["answer_correctness"] is not None or row["citation_faithfulness"] is not None for row in rows):
        raise ValueError("Quality scores require separate judgments")
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if (metadata["scientific_comparison"] or metadata["run_kind"] != "fixture_smoke"
            or metadata["run_status"] != "complete" or metadata["measured"] != len(rows)):
        raise ValueError("Incorrect development run metadata")
    return {"development_runs": len(rows), "questions": 12,
            "configurations": len(CONFIGURATIONS), "source_sha256": metadata["source_sha256"],
            "run_kind": "fixture_smoke", "scientific_comparison": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save results in a new or empty directory")
    parser.add_argument("--verbose", action="store_true", help="List every test")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or later is required")
    if args.output and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    groups = test_groups(suite)
    if not groups:
        parser.exit(1, "No tests found\n")
    print("Checking the agent and evaluation code...", flush=True)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if not result.wasSuccessful():
        return 1
    print("Running the development evaluation...", flush=True)
    try:
        if args.output:
            summary = asyncio.run(check_evaluation(args.output))
        else:
            with tempfile.TemporaryDirectory(prefix="omnistream-check-") as directory:
                summary = asyncio.run(check_evaluation(Path(directory) / "run"))
    except Exception as error:
        parser.exit(1, f"Evaluation check failed: {error}\n")
    summary.update({"tests_passed": result.testsRun, "test_groups": dict(groups),
                    "python": sys.version.split()[0],
                    "checked_at": datetime.now(timezone.utc).isoformat()})
    if args.output:
        (args.output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: {result.testsRun} tests and {summary['development_runs']} development runs.")
    print("The development run uses scripted components; live integrations are pending.")
    if args.output:
        print(f"Results saved to {args.output}")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as stream:
            stream.write(f"Passed **{result.testsRun} tests** and **{summary['development_runs']} development runs**.\n\n")
            stream.write("Scripted components and fictional data were used for this check.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
