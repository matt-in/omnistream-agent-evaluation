import csv
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from omnistream.contracts import CONFIGURATIONS, Claim, Evidence, ProviderBlocked, ProviderUnavailable, TimeScope
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

    async def test_failed_cases_are_retained_without_quality_measurements(self):
        class BrokenAdapter:
            async def answer(self, *args):
                raise ConnectionError("missing backend")
        rows = await EvaluationHarness(BrokenAdapter(), "fixture_smoke").run(SNAPSHOT, self.output, CONFIGURATIONS[:1])
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["status"] == "adapter_error" for r in rows))
        self.assertTrue(all(r["recall_at_10"] is None and not r["measured"] for r in rows))

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

    async def test_credit_failure_stops_and_resume_preserves_measured_cases_and_ratings(self):
        class InterruptibleAdapter:
            def __init__(self):
                self.fixture = FixtureSystem(SNAPSHOT / "items.jsonl")
                self.calls = []
                self.block = True

            async def answer(self, case, config):
                self.calls.append(case.id)
                if self.block and len(self.calls) == 3:
                    raise ProviderBlocked("Insufficient credit")
                return await self.fixture.answer(case, config)

        adapter = InterruptibleAdapter()
        harness = EvaluationHarness(adapter, "integrated", adapter_revision="test-settings")
        cfg = CONFIGURATIONS[:1]
        partial = await harness.run(SNAPSHOT, self.output, cfg)
        self.assertEqual(sum(r["measured"] for r in partial), 2)
        interrupted = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((interrupted["run_status"], interrupted["unmeasured"], interrupted["pending"]),
                         ("interrupted", 1, 9))
        path = self.output / "manual_ratings.csv"
        with path.open(encoding="utf-8") as stream:
            grades = list(csv.DictReader(stream))
        rated = grades[0]["question_id"]
        grades[0].update(answer_correctness="2", rater="test")
        EvaluationHarness._csv(path, grades)
        first_two = set(adapter.calls[:2])
        adapter.calls.clear()
        adapter.block = False
        completed = await harness.run(SNAPSHOT, self.output, cfg, resume=True)
        self.assertEqual(len(adapter.calls), 10)
        self.assertFalse(first_two & set(adapter.calls))
        self.assertTrue(all(r["measured"] for r in completed))
        with path.open(encoding="utf-8") as stream:
            grades = {r["question_id"]: r for r in csv.DictReader(stream)}
        self.assertEqual(grades[rated]["answer_correctness"], "2")
        self.assertEqual(json.loads((self.output / "run.json").read_text(encoding="utf-8"))["run_status"],
                         "complete")

    async def test_resume_rejects_different_adapter_revision(self):
        adapter = FixtureSystem(SNAPSHOT / "items.jsonl")
        cfg = CONFIGURATIONS[:1]
        await EvaluationHarness(adapter, "integrated", adapter_revision="first").run(SNAPSHOT, self.output, cfg)
        with self.assertRaisesRegex(ValueError, "Checkpoint does not match"):
            await EvaluationHarness(adapter, "integrated", adapter_revision="changed").run(
                SNAPSHOT, self.output, cfg, resume=True)

    async def test_retryable_failures_are_unmeasured_and_other_cases_continue(self):
        class UnavailableAdapter:
            def __init__(self):
                self.fixture = FixtureSystem(SNAPSHOT / "items.jsonl")

            async def answer(self, case, config):
                if case.id == "P01":
                    raise ProviderUnavailable("Rate limited")
                return await self.fixture.answer(case, config)

        rows = await EvaluationHarness(UnavailableAdapter(), "fixture_smoke").run(
            SNAPSHOT, self.output, CONFIGURATIONS[:1])
        failed = next(r for r in rows if r["question_id"] == "P01")
        self.assertEqual(failed["status"], "provider_unavailable")
        self.assertIsNone(failed["recall_at_10"])
        with (self.output / "comparison.csv").open(encoding="utf-8") as stream:
            full = next(r for r in csv.DictReader(stream) if r["kind"] == "all")
        self.assertEqual((full["expected"], full["measured"], full["failed"], full["recall_at_10_n"]),
                         ("12", "11", "1", "10"))

    async def test_retrieval_ranking_is_scored_separately_from_answer_context(self):
        _, questions, labels = load_snapshot(SNAPSHOT)
        question = questions[0]
        ids = tuple(labels[question["id"]])

        class RankingAdapter:
            def __init__(self):
                self.fixture = FixtureSystem(SNAPSHOT / "items.jsonl")

            async def answer(self, case, config):
                result = await self.fixture.answer(case, config)
                if case.id == question["id"]:
                    return replace(result, evidence=(), claims=(), citations=(), ranked_evidence_ids=ids)
                return result

        rows = await EvaluationHarness(RankingAdapter(), "fixture_smoke").run(
            SNAPSHOT, self.output, CONFIGURATIONS[:1])
        scored = next(r for r in rows if r["question_id"] == question["id"])
        self.assertEqual(scored["recall_at_10"], 1)
        self.assertIsNone(scored["citation_validity"])

    async def test_paired_correctness_uses_only_shared_question_grades(self):
        await self.create_run()
        path = self.output / "manual_ratings.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        grades = {
            ("own_agents", "P01"): 2, ("classic_agents", "P01"): 0,
            ("own_agents", "P02"): 2, ("classic_agents", "P02"): 1,
            ("classic_agents", "P03"): 2,
        }
        for row in rows:
            key = (row["configuration"], row["question_id"])
            if key in grades:
                row.update(answer_correctness=str(grades[key]), rater="test")
        EvaluationHarness._csv(path, rows)
        scored = Path(self.temp.name) / "scored"
        score_run(self.output, path, self.output / "citation_ratings.csv", scored)
        with (scored / "paired_correctness.csv").open(encoding="utf-8") as stream:
            pair = next(r for r in csv.DictReader(stream) if
                        (r["configuration"], r["comparison"], r["kind"]) == ("own_agents", "classic_agents", "all"))
        self.assertEqual((pair["expected_pairs"], pair["scored_pairs"], pair["mean_difference"]),
                         ("12", "2", "1.5"))

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
