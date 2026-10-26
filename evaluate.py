from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from commerce_lite.catalog import load_tools
from commerce_lite.metrics import evaluate_predictions
from commerce_lite.retrieval import RetrievalEngine


def load_queries(limit: int | None = None) -> list[dict[str, Any]]:
    queries = []
    for filename in ("evaluation_queries.json", "manual_queries.json"):
        with (ROOT / "data" / filename).open(encoding="utf-8") as source:
            queries.extend(json.load(source))
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError("The evaluation set is empty")
    return queries


def run_method(engine: RetrievalEngine, queries: list[dict[str, Any]], method: str) -> dict[str, Any]:
    engine.rank(queries[0]["query"], method=method, top_k=3)
    rankings: list[list[str]] = []
    truths: list[list[str]] = []
    latencies: list[float] = []
    for row in queries:
        started = time.perf_counter()
        ranked = engine.rank(row["query"], method=method, top_k=3)
        latencies.append((time.perf_counter() - started) * 1000)
        rankings.append([tool["name"] for tool in ranked])
        truths.append(row["expected_tools"])
    metrics = evaluate_predictions(
        rankings,
        truths,
        latencies,
        labels=[tool["name"] for tool in engine.tools],
    )
    metrics["method"] = method
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare e-commerce tool-routing retrieval methods")
    parser.add_argument("--methods", default="bm25", help="Comma-separated: bm25,dense,rrf,rrf_rerank")
    parser.add_argument("--limit", type=int, help="Limit examples for a quick smoke test")
    parser.add_argument("--output", default="results/evaluation_results.json")
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    valid_methods = {"bm25", "dense", "rrf", "rrf_rerank"}
    if not methods or set(methods) - valid_methods:
        parser.error("--methods must contain one or more of: bm25,dense,rrf,rrf_rerank")

    queries = load_queries(args.limit)
    engine = RetrievalEngine(load_tools())
    results = [run_method(engine, queries, method) for method in methods]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
