"""Frozen-snapshot evaluator. Adapter input excludes gold answers and qrels."""

import asyncio
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Protocol

from .agents import Answer
from .checkpoints import open_checkpoint, save_outcome, saved_outcomes
from .contracts import (
    CONFIGURATIONS, Configuration, ProviderBlocked, ProviderUnavailable, Query, TimeScope, timestamp,
)
from .metrics import citation_metrics, paired_bootstrap, retrieval_metrics, routing_metrics


@dataclass(frozen=True)
class EvaluationCase:
    """Unlabeled public input; only this object is given to a system adapter."""
    id: str
    kind: str
    query: Query


class SystemAdapter(Protocol):
    async def answer(self, case: EvaluationCase, config: Configuration) -> Answer: ...


def load_snapshot(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("items.jsonl", "questions.jsonl", "qrels.json"):
        expected = manifest["sha256"][filename]
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {filename}")
    questions = [json.loads(line) for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not questions:
        raise ValueError("A snapshot requires at least one question")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate evaluation question ID")
    qrels = json.loads((root / "qrels.json").read_text(encoding="utf-8"))
    for q in questions:
        timestamp(q["known_at"])
        if q.get("scope") is not None:
            TimeScope(**q["scope"])
        if not q["required_supports"] and q["answerable"]:
            raise ValueError("Answerable questions need labeled evidence")
        if q["id"] not in qrels:
            raise ValueError("Question is missing its relevance judgments")
        covered = set().union(*(set(v) for v in qrels[q["id"]].values()))
        if set(q["required_supports"]) - covered:
            raise ValueError("Missing supporting-evidence judgments")
    return manifest, questions, qrels


def _mean(values):
    available = [x for x in values if x is not None]
    return mean(available) if available else None


def _paired_correctness(rows, summary):
    scores, expected = {}, {}
    for row in rows:
        if row["answer_correctness"] is not None:
            for kind in ("all", row["kind"]):
                scores.setdefault((row["configuration"], kind), {})[row["question_id"]] = row["answer_correctness"]
    for row in summary:
        expected[row["configuration"], row["kind"]] = int(row.get("expected") or row["n"])
    pairs = (
        ("own_agents", "classic"), ("own_agents", "external"), ("own_agents", "classic_agents"),
        ("own_agents", "own"), ("classic_agents", "classic"), ("own", "classic"),
    )
    results = []
    for left, right in pairs:
        kinds = sorted(kind for config, kind in expected if config == left and (right, kind) in expected)
        for kind in kinds:
            a, b = scores.get((left, kind), {}), scores.get((right, kind), {})
            common = sorted(a.keys() & b.keys())
            x, y = [a[key] for key in common], [b[key] for key in common]
            interval = paired_bootstrap(x, y) if len(common) >= 2 else {}
            results.append({
                "configuration": left, "comparison": right, "kind": kind,
                "expected_pairs": min(expected[left, kind], expected[right, kind]),
                "scored_pairs": len(common), "mean_difference": _mean([l - r for l, r in zip(x, y)]),
                "lower_95": interval.get("lower_95"), "upper_95": interval.get("upper_95"),
            })
    return results


class EvaluationHarness:
    MEASURED_STATUSES = frozenset({"ok", "partial", "abstained", "no_evidence", "no_bubbles"})
    METRICS = (
        "recall_at_10", "mrr", "hits_at_1", "support_complete_at_10",
        "routing_precision", "routing_recall", "citation_validity", "citation_coverage",
        "answer_correctness", "citation_faithfulness", "changed_fact_correct",
    )

    def __init__(self, adapter: SystemAdapter, run_kind: str, seed=20261113,
                 query_timeout_seconds=240, *, adapter_revision: str | None = None):
        if run_kind not in {"fixture_smoke", "integrated"}:
            raise ValueError("Run kind must identify fixture or integrated execution")
        if not math.isfinite(query_timeout_seconds) or query_timeout_seconds <= 0:
            raise ValueError("Query timeout must be positive and finite")
        self.adapter, self.run_kind, self.seed = adapter, run_kind, seed
        self.query_timeout_seconds = query_timeout_seconds
        self.adapter_revision = adapter_revision

    async def run(self, snapshot, output, configs=CONFIGURATIONS, *, resume=False):
        output = Path(output)
        if not resume and output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory must be new or empty")
        if resume and self.run_kind == "integrated" and not self.adapter_revision:
            raise ValueError("Integrated resume requires a recorded adapter revision")
        configs = tuple(configs)
        if not configs or len({c.name for c in configs}) != len(configs):
            raise ValueError("The run requires distinct configuration names")
        manifest, questions, qrels = load_snapshot(snapshot)
        source_dir = Path(__file__).parent
        code_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                                           for p in sorted(source_dir.glob("*.py")))).hexdigest()
        identity = {
            "run_kind": self.run_kind, "snapshot": manifest, "seed": self.seed,
            "source_sha256": code_hash, "adapter_revision": self.adapter_revision,
            "adapter_class": type(self.adapter).__module__ + "." + type(self.adapter).__qualname__,
            "configurations": [asdict(c) for c in configs],
            "query_timeout_seconds": self.query_timeout_seconds,
        }
        order = [(q, c) for q in questions for c in configs]
        random.Random(self.seed).shuffle(order)
        output.mkdir(parents=True, exist_ok=True)
        interrupted = False
        with open_checkpoint(output / "checkpoints.sqlite3", identity, resume=resume) as checkpoint:
            completed = {(r["question_id"], r["configuration"])
                         for r, _ in saved_outcomes(checkpoint) if r["measured"]}
            for question, config in order:
                if (question["id"], config.name) in completed:
                    continue
                row, trace = await self._run_case(question, config, qrels[question["id"]])
                save_outcome(checkpoint, row, trace)
                if row["status"] == "provider_blocked":
                    interrupted = True
                    break
            outcomes = saved_outcomes(checkpoint)
        rows = [row for row, _ in outcomes]
        traces = [trace for _, trace in outcomes]
        self._export(output, configs, questions, rows, traces)
        measured = sum(row["measured"] for row in rows)
        state = "interrupted" if interrupted else ("complete" if measured == len(order) else "incomplete")
        metadata = {
            **identity, "run_status": state, "expected": len(order), "attempted": len(rows),
            "measured": measured, "unmeasured": len(rows) - measured, "pending": len(order) - len(rows),
            "python": sys.version, "platform": platform.platform(),
            "configuration_order": [(q["id"], c.name) for q, c in order],
            "question_count": len(questions), "configuration_count": len(configs),
        }
        if self.run_kind == "fixture_smoke":
            metadata["scientific_comparison"] = False
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
        return rows

    async def _run_case(self, question, config, qrels):
        scope = TimeScope(**question["scope"]) if question.get("scope") is not None else None
        public = EvaluationCase(question["id"], question["kind"], Query(
            question["question"], question["known_at"], scope))
        started = time.perf_counter()
        answer, error, status = None, None, "adapter_error"
        try:
            result = await asyncio.wait_for(self.adapter.answer(public, config), self.query_timeout_seconds)
            if not isinstance(result, Answer):
                raise TypeError("Adapter must return an Answer")
            answer = result
            status = answer.status
            if answer.errors:
                error = answer.errors[0]
        except ProviderBlocked as exc:
            status, error = "provider_blocked", type(exc).__name__
        except ProviderUnavailable as exc:
            status, error = "provider_unavailable", type(exc).__name__
        except Exception as exc:
            error = type(exc).__name__
        elapsed_ms = (time.perf_counter() - started) * 1000
        measured = status in self.MEASURED_STATUSES
        metrics = dict.fromkeys(self.METRICS)
        if measured:
            evidence_ids = [e.unit_id for e in answer.evidence]
            ranking = answer.ranked_evidence_ids if answer.ranked_evidence_ids is not None else evidence_ids
            selected = answer.route.bubbles if answer.route else ()
            metrics.update(retrieval_metrics(ranking, qrels, question["required_supports"]))
            metrics.update(routing_metrics(selected, question["required_bubbles"]))
            metrics.update(citation_metrics(answer.claims, evidence_ids))
        row = {
            "question_id": question["id"], "kind": question["kind"], "configuration": config.name,
            "run_kind": self.run_kind, "status": status, "error": error, "measured": measured,
            **metrics, "freshness_lag_seconds": None, "latency_ms": elapsed_ms,
            "tool_calls": sum(len(i.history) for i in answer.investigations) if answer else 0,
            "tokens": None, "cost": None, "peak_total_memory_mb": None,
        }
        trace = {"question_id": public.id, "configuration": config.name,
                 "run_kind": self.run_kind, "query": asdict(public.query),
                 "answer": asdict(answer) if answer else None, "status": status, "error": error,
                 "measured": measured}
        return row, trace

    def _export(self, output, configs, questions, rows, traces):
        summary = []
        for config in configs:
            for kind in ("all", *sorted({q["kind"] for q in questions})):
                selected = [r for r in rows if r["configuration"] == config.name and (kind == "all" or r["kind"] == kind)]
                expected = sum(kind == "all" or q["kind"] == kind for q in questions)
                measured = sum(r["measured"] for r in selected)
                summary.append({"configuration": config.name, "kind": kind, "run_kind": self.run_kind,
                                "n": len(selected), "expected": expected, "measured": measured,
                                "failed": len(selected) - measured, "pending": expected - len(selected),
                                **{m: _mean([r[m] for r in selected]) for m in self.METRICS},
                                **{m + "_n": sum(r[m] is not None for r in selected) for m in self.METRICS}})
        self._csv(output / "per_question.csv", rows)
        self._csv(output / "comparison.csv", summary)
        with (output / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for trace in traces:
                stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
        ratings = [{"question_id": r["question_id"], "configuration": r["configuration"],
                    "answer_correctness": "", "changed_fact_correct": "", "rater": ""} for r in rows if r["measured"]]
        self._rating_sheet(output / "manual_ratings.csv", ratings, ("question_id", "configuration"),
                           ("answer_correctness", "changed_fact_correct", "rater"))
        citation_ratings = []
        for trace in traces:
            if trace["measured"] and trace["answer"]:
                for i, claim in enumerate(trace["answer"]["claims"]):
                    for unit in dict.fromkeys(claim["evidence_ids"]):
                        citation_ratings.append({"question_id": trace["question_id"],
                                                 "configuration": trace["configuration"],
                                                 "claim_index": i, "unit_id": unit,
                                                 "supported": "", "rater": ""})
        self._rating_sheet(output / "citation_ratings.csv", citation_ratings,
                           ("question_id", "configuration", "claim_index", "unit_id"), ("supported", "rater"))

    @classmethod
    def _rating_sheet(cls, path, rows, keys, labels):
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as stream:
                previous = {}
                for entry in csv.DictReader(stream):
                    identity = tuple(entry[k] for k in keys)
                    if identity in previous:
                        raise ValueError("Duplicate entry in existing rating sheet")
                    previous[identity] = {label: entry[label] for label in labels}
            for row in rows:
                identity = tuple(str(row[k]) for k in keys)
                row.update(previous.get(identity, {}))
        cls._csv(path, rows, (*keys, *labels))

    @staticmethod
    def _csv(path, rows, fields=None):
        if not rows and fields is None:
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def score_run(run_directory, correctness_file, citation_file, output_directory):
    """Import independently assigned labels without inventing missing judgments."""
    root, output = Path(run_directory), Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh scoring output directory")
    with (root / "per_question.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "comparison.csv").open(encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    traces = [json.loads(line) for line in (root / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {}
    unmeasured = set()
    for trace in traces:
        key = (trace["question_id"], trace["configuration"])
        if trace.get("measured") is False:
            unmeasured.add(key)
            expected[key] = set()
            continue
        expected[key] = {(i, u) for i, claim in enumerate(trace["answer"]["claims"] if trace["answer"] else [])
                         for u in claim["evidence_ids"]}
    grades, entailments = {}, {}
    with Path(correctness_file).open(encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            key = (entry["question_id"], entry["configuration"])
            if key not in expected or key in grades:
                raise ValueError("Unknown or duplicate correctness key")
            grade = entry["answer_correctness"].strip()
            changed = entry.get("changed_fact_correct", "").strip()
            if key in unmeasured and (grade or changed):
                raise ValueError("Unmeasured executions cannot receive answer ratings")
            if grade not in ("", "0", "1", "2") or changed not in ("", "0", "1"):
                raise ValueError("Correctness must be 0/1/2; changed-fact label must be 0/1")
            if (grade or changed) and not entry.get("rater", "").strip():
                raise ValueError("Every assigned grade needs a rater or judge version")
            grades[key] = (int(grade) if grade else None, int(changed) if changed else None)
    with Path(citation_file).open(encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            key = (entry["question_id"], entry["configuration"])
            pair = (int(entry["claim_index"]), entry["unit_id"])
            if key not in expected or pair not in expected[key]:
                raise ValueError("Unknown claim/citation judgment")
            labels = entailments.setdefault(key, {})
            if pair in labels:
                raise ValueError("Duplicate citation judgment")
            label = entry["supported"].strip()
            if label not in ("", "0", "1"):
                raise ValueError("Citation entailment must be 0/1")
            if label and not entry.get("rater", "").strip():
                raise ValueError("Citation judgments need a rater or judge version")
            labels[pair] = int(label) if label else None
    for row in rows:
        key = (row["question_id"], row["configuration"])
        row["answer_correctness"], row["changed_fact_correct"] = grades.get(key, (None, None))
        labels = entailments.get(key, {})
        judged = [labels.get(pair) for pair in expected[key]]
        row["citation_judgments_n"] = sum(v is not None for v in judged)
        row["citation_judgments_expected"] = len(judged)
        row["citation_faithfulness"] = mean(judged) if judged and all(v is not None for v in judged) else None
    for row in summary:
        selected = [r for r in rows if r["configuration"] == row["configuration"] and (row["kind"] == "all" or r["kind"] == row["kind"])]
        for metric in ("answer_correctness", "citation_faithfulness", "changed_fact_correct"):
            row[metric] = _mean([r[metric] for r in selected])
            row[metric + "_n"] = sum(r[metric] is not None for r in selected)
    output.mkdir(parents=True, exist_ok=True)
    EvaluationHarness._csv(output / "per_question.csv", rows)
    EvaluationHarness._csv(output / "comparison.csv", summary)
    EvaluationHarness._csv(output / "paired_correctness.csv", _paired_correctness(rows, summary))
    meta = json.loads((root / "run.json").read_text(encoding="utf-8"))
    meta["scoring_inputs_sha256"] = {"correctness": hashlib.sha256(Path(correctness_file).read_bytes()).hexdigest(),
                                       "citations": hashlib.sha256(Path(citation_file).read_bytes()).hexdigest(),
                                       "traces": hashlib.sha256((root / "traces.jsonl").read_bytes()).hexdigest()}
    (output / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return rows
