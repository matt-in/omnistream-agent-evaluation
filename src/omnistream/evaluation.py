"""Frozen-snapshot evaluator. Adapter input excludes gold answers and qrels."""

import asyncio
import csv
import hashlib
import json
import random
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Protocol

from .agents import Answer
from .contracts import CONFIGURATIONS, Configuration, Query, TimeScope, timestamp
from .metrics import citation_metrics, retrieval_metrics, routing_metrics


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
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate evaluation question ID")
    qrels = json.loads((root / "qrels.json").read_text(encoding="utf-8"))
    for q in questions:
        timestamp(q["known_at"])
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


class EvaluationHarness:
    def __init__(self, adapter: SystemAdapter, run_kind: str, seed=20261113,
                 query_timeout_seconds=240):
        if run_kind not in {"fixture_smoke", "integrated"}:
            raise ValueError("Run kind must identify fixture or integrated execution")
        self.adapter, self.run_kind, self.seed = adapter, run_kind, seed
        self.query_timeout_seconds = query_timeout_seconds

    async def run(self, snapshot, output, configs=CONFIGURATIONS):
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a fresh output directory to preserve run evidence")
        manifest, questions, qrels = load_snapshot(snapshot)
        output.mkdir(parents=True, exist_ok=True)
        order = [(q, c) for q in questions for c in configs]
        random.Random(self.seed).shuffle(order)
        rows, traces = [], []
        for question, config in order:
            public = EvaluationCase(question["id"], question["kind"], Query(
                question["question"], question["known_at"], TimeScope(**question["scope"])))
            started = time.perf_counter()
            answer, error = None, None
            try:
                answer = await asyncio.wait_for(self.adapter.answer(public, config), self.query_timeout_seconds)
                if not isinstance(answer, Answer):
                    raise TypeError("Adapter must return an Answer")
            except Exception as exc:
                answer = None
                error = type(exc).__name__
            elapsed_ms = (time.perf_counter() - started) * 1000
            evidence_ids = [e.unit_id for e in answer.evidence] if answer else []
            selected = answer.route.bubbles if answer and answer.route else ()
            row = {"question_id": question["id"], "kind": question["kind"],
                   "configuration": config.name, "run_kind": self.run_kind,
                   "status": answer.status if answer else "adapter_error", "error": error,
                   **retrieval_metrics(evidence_ids, qrels[question["id"]], question["required_supports"]),
                   **routing_metrics(selected, question["required_bubbles"]),
                   **citation_metrics(answer.claims if answer else (), evidence_ids),
                   "answer_correctness": None, "citation_faithfulness": None,
                   "changed_fact_correct": None, "freshness_lag_seconds": None,
                   "latency_ms": elapsed_ms,
                   "tool_calls": sum(len(i.history) for i in answer.investigations) if answer else 0,
                   "tokens": None, "cost": None, "peak_total_memory_mb": None}
            rows.append(row)
            traces.append({"question_id": public.id, "configuration": config.name,
                           "run_kind": self.run_kind, "query": asdict(public.query),
                           "answer": asdict(answer) if answer else None, "error": error})
        rows.sort(key=lambda r: (r["configuration"], r["question_id"]))
        metrics = ("recall_at_10", "mrr", "hits_at_1", "routing_precision", "routing_recall",
                   "citation_validity", "citation_coverage", "answer_correctness", "citation_faithfulness")
        summary = []
        for config in configs:
            for kind in ("all", *sorted({q["kind"] for q in questions})):
                selected = [r for r in rows if r["configuration"] == config.name and (kind == "all" or r["kind"] == kind)]
                summary.append({"configuration": config.name, "kind": kind, "run_kind": self.run_kind,
                                "n": len(selected), "failed": sum(r["status"] not in ("ok", "partial", "abstained", "no_evidence") for r in selected),
                                **{m: _mean([r[m] for r in selected]) for m in metrics},
                                **{m + "_n": sum(r[m] is not None for r in selected) for m in metrics}})
        self._csv(output / "per_question.csv", rows)
        self._csv(output / "comparison.csv", summary)
        with (output / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for trace in traces:
                stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
        ratings = [{"question_id": r["question_id"], "configuration": r["configuration"],
                    "answer_correctness": "", "changed_fact_correct": "", "rater": ""} for r in rows]
        self._csv(output / "manual_ratings.csv", ratings)
        citation_ratings = []
        for trace in traces:
            if trace["answer"]:
                for i, claim in enumerate(trace["answer"]["claims"]):
                    for unit in dict.fromkeys(claim["evidence_ids"]):
                        citation_ratings.append({"question_id": trace["question_id"],
                                                 "configuration": trace["configuration"],
                                                 "claim_index": i, "unit_id": unit,
                                                 "supported": "", "rater": ""})
        self._csv(output / "citation_ratings.csv", citation_ratings)
        source_dir = Path(__file__).parent
        code_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(source_dir.glob("*.py")))).hexdigest()
        metadata = {"run_kind": self.run_kind, "scientific_comparison": self.run_kind == "integrated",
                    "snapshot": manifest, "seed": self.seed,
                    "source_sha256": code_hash, "python": sys.version, "platform": platform.platform(),
                    "adapter_class": type(self.adapter).__module__ + "." + type(self.adapter).__qualname__,
                    "configuration_order": [(q["id"], c.name) for q, c in order],
                    "question_count": len(questions), "configuration_count": len(configs),
                    "query_timeout_seconds": self.query_timeout_seconds,
                    "note": "Fixture scores test software plumbing and cannot demonstrate GraphRAG gains."
                    if self.run_kind == "fixture_smoke" else "Inspect adapter provenance and judgment coverage before interpreting results."}
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
        return rows

    @staticmethod
    def _csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    for trace in traces:
        key = (trace["question_id"], trace["configuration"])
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
        for metric in ("answer_correctness", "citation_faithfulness"):
            row[metric] = _mean([r[metric] for r in selected])
            row[metric + "_n"] = sum(r[metric] is not None for r in selected)
    output.mkdir(parents=True, exist_ok=True)
    EvaluationHarness._csv(output / "per_question.csv", rows)
    EvaluationHarness._csv(output / "comparison.csv", summary)
    meta = json.loads((root / "run.json").read_text(encoding="utf-8"))
    meta["scoring_inputs_sha256"] = {"correctness": hashlib.sha256(Path(correctness_file).read_bytes()).hexdigest(),
                                       "citations": hashlib.sha256(Path(citation_file).read_bytes()).hexdigest(),
                                       "traces": hashlib.sha256((root / "traces.jsonl").read_bytes()).hexdigest()}
    (output / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return rows
