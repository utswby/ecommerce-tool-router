from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from commerce_lite.catalog import load_tools
from commerce_lite.metrics import evaluate_predictions
from commerce_lite.retrieval import RetrievalEngine
from evaluate import load_queries


class KeywordEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "inventory" in lowered or "stock" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "return" in lowered or "exchange" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class OrderedReranker:
    def rerank(self, query, documents):
        return range(len(documents))


def test_catalog_and_benchmark_labels_are_consistent():
    tools = load_tools()
    names = {tool["name"] for tool in tools}
    rows = load_queries()

    assert len(tools) == 8
    assert len(rows) == 96
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["language"] == "en" for row in rows)
    assert all(set(row["expected_tools"]) <= names for row in rows)
    assert any(len(row["expected_tools"]) > 1 for row in rows)


def test_bm25_routes_inventory_queries_to_stock_tool():
    engine = RetrievalEngine(load_tools())

    ranked = engine.rank("How many units are in inventory?", method="bm25", top_k=1)

    assert ranked[0]["name"] == "get_stock"


def test_dense_retrieval_uses_injected_embedder_without_model_download():
    engine = RetrievalEngine(load_tools(), embedder=KeywordEmbedder())

    ranked = engine.rank("Check inventory for SKU-0042", method="dense", top_k=1)

    assert ranked[0]["name"] == "get_stock"


def test_reranker_scores_reorder_only_the_hybrid_shortlist():
    engine = RetrievalEngine(
        load_tools(),
        embedder=KeywordEmbedder(),
        reranker=OrderedReranker(),
    )

    query = "Check inventory for SKU-0042"
    hybrid = engine.rank(query, method="rrf", top_k=8)
    ranked = engine.rank(query, method="rrf_rerank", top_k=3)

    assert ranked[0]["name"] == hybrid[-1]["name"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_exact_set_metrics_use_label_cardinality_and_all_catalog_labels():
    result = evaluate_predictions(
        rankings=[["get_stock", "get_item_info"], ["get_cart_summary"]],
        truths=[["get_item_info", "get_stock"], ["get_cart_summary"]],
        latencies_ms=[1.0, 3.0],
        labels=["get_item_info", "get_stock", "get_cart_summary", "get_order_list"],
    )

    assert result["exact_tool_set_accuracy"] == 1.0
    assert result["micro_f1"] == 1.0
    assert result["macro_f1"] == 0.75
    assert result["recall_at_1"] == 0.75
    assert result["mean_latency_ms"] == 2.0


def test_query_files_are_small_and_valid_json():
    for filename in ("evaluation_queries.json", "manual_queries.json", "tools.json"):
        with (ROOT / "data" / filename).open(encoding="utf-8") as source:
            assert isinstance(json.load(source), list)
