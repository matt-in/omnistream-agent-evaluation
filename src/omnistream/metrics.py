"""Metric definitions are explicit; missing judgments stay missing."""

import math
import random
from statistics import mean

from .contracts import timestamp


def retrieval_metrics(ranked_ids, qrels, required_supports, k=10):
    """Fact coverage in top k unique returned units, allowing chunk/fact alternatives.

    qrels: unit_id -> list of human-labeled canonical support assertion IDs.
    Never infer relevance from ID spelling or feed the qrels to the retriever.
    MRR is measured over the full ranked list. Hits@1 uses the first unit.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = list(dict.fromkeys(ranked_ids))
    required = set(required_supports)
    if not required:
        return {"recall_at_10": None, "mrr": None, "hits_at_1": None, "support_complete_at_10": None}
    covered = set().union(*(set(qrels.get(u, ())) for u in ranked[:k])) & required
    relevant = [bool(set(qrels.get(u, ())) & required) for u in ranked]
    first = next((i for i, hit in enumerate(relevant, 1) if hit), None)
    return {"recall_at_10": len(covered) / len(required),
            "mrr": 1 / first if first else 0.0,
            "hits_at_1": float(bool(relevant and relevant[0])),
            "support_complete_at_10": float(covered == required)}


def routing_metrics(predicted, required):
    predicted, required = set(predicted), set(required)
    matched = len(predicted & required)
    return {"routing_precision": matched / len(predicted) if predicted else 0.0,
            "routing_recall": matched / len(required) if required else None}


def citation_metrics(claims, retrieved_ids):
    """Structural validity and coverage only, NEVER semantic faithfulness."""
    allowed = set(retrieved_ids)
    references = [(i, u) for i, c in enumerate(claims) for u in dict.fromkeys(c.evidence_ids)]
    return {"citation_validity": sum(u in allowed for _, u in references) / len(references) if references else None,
            "citation_coverage": sum(bool(c.evidence_ids) for c in claims) / len(claims) if claims else None}


def faithfulness(supported_citations):
    """Input is independently judged claim/citation entailment (booleans)."""
    labels = list(supported_citations)
    if any(type(x) is not bool for x in labels):
        raise ValueError("Entailment labels must be booleans")
    return sum(labels) / len(labels) if labels else None


def freshness_lag(published_at, first_retrievable_at, poll_seconds):
    if poll_seconds <= 0:
        raise ValueError("Poll interval must be positive")
    lag = (timestamp(first_retrievable_at) - timestamp(published_at)).total_seconds()
    if lag < 0:
        raise ValueError("An item cannot be retrievable before publication")
    return {"freshness_lag_seconds": lag, "freshness_pass": lag <= poll_seconds + 300}


def exact_set_accuracy(predicted, gold):
    if len(predicted) != len(gold):
        raise ValueError("Labels must be paired")
    return mean(set(p) == set(g) for p, g in zip(predicted, gold)) if gold else None


def judge_agreement(judge, human):
    """Raw agreement and unweighted Cohen kappa on the 0/1/2 rubric."""
    if len(judge) != len(human) or not judge:
        raise ValueError("Calibration needs paired nonempty labels")
    if any(type(x) is not int or x not in (0, 1, 2) for x in [*judge, *human]):
        raise ValueError("Correctness labels must be 0, 1 or 2")
    observed = mean(a == b for a, b in zip(judge, human))
    expected = sum((judge.count(v) / len(judge)) * (human.count(v) / len(human)) for v in (0, 1, 2))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"agreement": observed, "cohen_kappa": kappa, "n": len(judge)}


def paired_bootstrap(left, right, iterations=2000, seed=20261113):
    """Percentile interval for paired mean differences; positive favors left."""
    if len(left) != len(right) or not left or iterations < 100:
        raise ValueError("Need paired scores and at least 100 resamples")
    deltas = [a - b for a, b in zip(left, right)]
    if any(not math.isfinite(d) for d in deltas):
        raise ValueError("Scores must be finite")
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(iterations))
    return {"mean_difference": mean(deltas), "lower_95": draws[int(iterations * .025)],
            "upper_95": draws[min(iterations - 1, int(iterations * .975))],
            "n": len(deltas), "seed": seed}
