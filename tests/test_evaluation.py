import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from omnistream.contracts import CONFIGURATIONS, Claim, Evidence, TimeScope
from omnistream.evaluation import EvaluationHarness, load_snapshot, score_run
from omnistream.fixtures import FixtureSystem
from omnistream.metrics import (
    citation_metrics, exact_set_accuracy, faithfulness, freshness_lag,
    judge_agreement, paired_bootstrap, retrieval_metrics, routing_metrics,
)


SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "pilot_snapshot"


class MetricTests(unittest.TestCase):
    def test_fact_coverage_deduplicates_units_and_equivalent_passages(self):
        metrics = retrieval_metrics(["chunk", "chunk", "fact", "missing"],
                                    {"chunk": ["a", "b"], "fact": ["a"]}, ["a", "b", "c"])
        self.assertAlmostEqual(metrics["recall_at_10"], 2 / 3)
        self.assertEqual(metrics["hits_at_1"], 1)
        self.assertEqual(metrics["support_complete_at_10"], 0)

    def test_mrr_and_unanswerable_cases(self):
        result = retrieval_metrics(["noise", "good"], {"good": ["a"]}, ["a"])
        self.assertEqual(result["mrr"], .5)
        self.assertEqual(result["hits_at_1"], 0)
        self.assertIsNone(retrieval_metrics([], {}, [])["recall_at_10"])
        self.assertEqual(retrieval_metrics([], {}, ["a"])["mrr"], 0)

    def test_relevance_comes_only_from_labels(self):
        self.assertEqual(retrieval_metrics(["a"], {}, ["a"])["recall_at_10"], 0)

    def test_routing_is_multilabel_set_precision_and_recall(self):
        result = routing_metrics(["news", "finance", "news"], ["finance", "research"])
        self.assertEqual(result, {"routing_precision": .5, "routing_recall": .5})
        self.assertEqual(routing_metrics([], ["news"])["routing_recall"], 0)

    def test_citation_structure_is_separate_from_faithfulness(self):
        result = citation_metrics([Claim("Unsupported meaning", ("real_id",))], ["real_id"])
        self.assertEqual(result["citation_validity"], 1)
        self.assertEqual(faithfulness([True, False]), .5)
        self.assertIsNone(faithfulness([]))
        self.assertIsNone(citation_metrics([], [])["citation_validity"])

    def test_freshness_uses_publication_to_searchable_time(self):
        result = freshness_lag("2026-11-01T00:00:00Z", "2026-11-01T00:20:01Z", 900)
        self.assertFalse(result["freshness_pass"])
        with self.assertRaises(ValueError):
            freshness_lag("2026-11-02T00:00:00Z", "2026-11-01T00:00:00Z", 900)

    def test_exact_set_ingestion_accuracy(self):
        self.assertEqual(exact_set_accuracy([["a", "b"], ["a"]], [["b", "a"], ["b"]]), .5)

    def test_judge_calibration_requires_paired_valid_ratings(self):
        result = judge_agreement([0, 1, 2], [0, 1, 2])
        self.assertEqual(result["cohen_kappa"], 1)
        self.assertIsNone(judge_agreement([2, 2], [2, 2])["cohen_kappa"])
        with self.assertRaises(ValueError):
            judge_agreement([3], [2])

    def test_bootstrap_pairs_are_reproducible(self):
        one = paired_bootstrap([2, 1, 0], [1, 1, 0], iterations=300)
        two = paired_bootstrap([2, 1, 0], [1, 1, 0], iterations=300)
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["mean_difference"], 1 / 3)


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "run"

    async def create_run(self):
        return await EvaluationHarness(FixtureSystem(SNAPSHOT / "items.jsonl"), "fixture_smoke").run(SNAPSHOT, self.output)

    async def test_six_configuration_smoke_run_has_no_invented_quality_scores(self):
        rows = await self.create_run()
        self.assertEqual(len(rows), 72)
        for config in CONFIGURATIONS:
            self.assertEqual(sum(r["configuration"] == config.name for r in rows), 12)
        self.assertTrue(all(r["answer_correctness"] is None and r["citation_faithfulness"] is None for r in rows))
        self.assertTrue(all(r["peak_total_memory_mb"] is None and r["cost"] is None for r in rows))
        metadata = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["scientific_comparison"])
        self.assertEqual(metadata["run_kind"], "fixture_smoke")

    async def test_every_trace_respects_known_time_and_validity(self):
        await self.create_run()
        for line in (self.output / "traces.jsonl").read_text(encoding="utf-8").splitlines():
            trace = json.loads(line)
            for raw in trace["answer"]["evidence"]:
                evidence = Evidence(**raw)
                self.assertTrue(evidence.visible(TimeScope(**trace["query"]["scope"]), trace["query"]["known_at"]))
            if trace["question_id"] == "P10":
                self.assertNotIn("f09", [e["unit_id"] for e in trace["answer"]["evidence"]])

    async def test_snapshot_hash_mismatch_is_rejected(self):
        copied = Path(self.temp.name) / "snapshot"
        shutil.copytree(SNAPSHOT, copied)
        with (copied / "questions.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_snapshot(copied)

    async def test_adapter_input_excludes_answers_and_relevance_labels(self):
        class CheckingAdapter:
            async def answer(inner, case, config):
                self.assertEqual(set(vars(case)), {"id", "kind", "query"})
                self.assertFalse(hasattr(case, "reference_answer"))
                return await FixtureSystem(SNAPSHOT / "items.jsonl").answer(case, config)
        await EvaluationHarness(CheckingAdapter(), "fixture_smoke").run(SNAPSHOT, self.output, CONFIGURATIONS[:1])

    async def test_failed_cases_are_retained_in_denominator(self):
        class BrokenAdapter:
            async def answer(self, *args):
                raise ConnectionError("missing backend")
        rows = await EvaluationHarness(BrokenAdapter(), "fixture_smoke").run(SNAPSHOT, self.output, CONFIGURATIONS[:1])
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["status"] == "adapter_error" for r in rows))
        self.assertEqual(next(r for r in rows if r["question_id"] == "P01")["recall_at_10"], 0)

    async def test_invalid_adapter_response_is_recorded_for_each_case(self):
        class InvalidAdapter:
            async def answer(self, *args):
                return {"text": "Unexpected response"}

        rows = await EvaluationHarness(InvalidAdapter(), "fixture_smoke").run(
            SNAPSHOT, self.output, CONFIGURATIONS[:1])
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["status"] == "adapter_error" for r in rows))
        self.assertTrue(all(r["error"] == "TypeError" for r in rows))

    async def test_run_evidence_is_not_overwritten(self):
        await self.create_run()
        with self.assertRaises(FileExistsError):
            await self.create_run()

    async def test_empty_ratings_remain_unmeasured_after_import(self):
        await self.create_run()
        result = score_run(self.output, self.output / "manual_ratings.csv", self.output / "citation_ratings.csv",
                           Path(self.temp.name) / "scored")
        self.assertTrue(all(r["answer_correctness"] is None and r["citation_faithfulness"] is None for r in result))

    async def test_partial_citation_labels_do_not_become_full_faithfulness(self):
        await self.create_run()
        path = self.output / "citation_ratings.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["supported"], rows[0]["rater"] = "1", "unit-test-only"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = score_run(self.output, self.output / "manual_ratings.csv", path, Path(self.temp.name) / "scored")
        affected = next(r for r in result if r["question_id"] == rows[0]["question_id"] and r["configuration"] == rows[0]["configuration"])
        if affected["citation_judgments_expected"] > 1:
            self.assertIsNone(affected["citation_faithfulness"])
        else:
            self.assertEqual(affected["citation_faithfulness"], 1)


if __name__ == "__main__":
    unittest.main()
