from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any


def evaluate_predictions(
    rankings: Sequence[Sequence[str]],
    truths: Sequence[Sequence[str]],
    latencies_ms: Sequence[float],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not (len(rankings) == len(truths) == len(latencies_ms)):
        raise ValueError("Rankings, labels, and latencies must have matching lengths")
    if not truths:
        raise ValueError("At least one evaluation example is required")

    exact = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    recall_at_1 = 0.0
    recall_at_3 = 0.0
    per_tool: dict[str, Counter[str]] = {}

    for ranking, expected_names in zip(rankings, truths):
        expected = set(expected_names)
        predicted = set(ranking[: len(expected)])
        exact += int(predicted == expected)
        true_positives += len(expected & predicted)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)
        recall_at_1 += len(expected & set(ranking[:1])) / len(expected)
        recall_at_3 += len(expected & set(ranking[:3])) / len(expected)
        for name in set(ranking) | expected:
            counts = per_tool.setdefault(name, Counter())
            counts["tp"] += int(name in predicted and name in expected)
            counts["fp"] += int(name in predicted and name not in expected)
            counts["fn"] += int(name not in predicted and name in expected)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    class_f1 = []
    for name in labels or per_tool:
        counts = per_tool.get(name, Counter())
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        class_f1.append(2 * counts["tp"] / denominator if denominator else 0.0)

    ordered_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)
    return {
        "examples": len(truths),
        "exact_tool_set_accuracy": exact / len(truths),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": statistics.mean(class_f1) if class_f1 else 0.0,
        "recall_at_1": recall_at_1 / len(truths),
        "recall_at_3": recall_at_3 / len(truths),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "p95_latency_ms": ordered_latencies[p95_index],
        "exact_match_cardinality": "top-ranked count equals number of labelled tools",
    }
