from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from rank_bm25 import BM25Okapi

from .catalog import tool_document


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    norm_left = math.sqrt(sum(value * value for value in left_values))
    norm_right = math.sqrt(sum(value * value for value in right_values))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class RetrievalEngine:
    def __init__(
        self,
        tools: list[dict[str, Any]],
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        if not tools:
            raise ValueError("At least one tool is required")
        self.tools = tools
        self.documents = [tool_document(tool) for tool in tools]
        self._bm25 = BM25Okapi([tokenize(document) for document in self.documents])
        self._embedder = embedder
        self._reranker = reranker
        self._document_vectors: list[list[float]] | None = None

    def _load_embedder(self) -> Any:
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError("Install fastembed from requirements.txt to use dense retrieval") from exc
            self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        return self._embedder

    def _load_reranker(self) -> Any:
        if self._reranker is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install fastembed from requirements.txt to use the reranker") from exc
            self._reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
        return self._reranker

    @staticmethod
    def _vectors(embedder: Any, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in embedder.embed(texts)]

    def _dense_scores(self, query: str) -> list[float]:
        embedder = self._load_embedder()
        if self._document_vectors is None:
            self._document_vectors = self._vectors(embedder, self.documents)
        query_text = "Represent this sentence for searching relevant passages: " + query
        query_vector = self._vectors(embedder, [query_text])[0]
        return [cosine(query_vector, vector) for vector in self._document_vectors]

    @staticmethod
    def _rank(scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda index: (-scores[index], index))

    def _bm25_order(self, query: str) -> tuple[list[int], list[float]]:
        scores = [float(score) for score in self._bm25.get_scores(tokenize(query))]
        return self._rank(scores), scores

    def _dense_order(self, query: str) -> tuple[list[int], list[float]]:
        scores = self._dense_scores(query)
        return self._rank(scores), scores

    def rank(self, query: str, method: str = "bm25", top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if method not in {"bm25", "dense", "rrf", "rrf_rerank"}:
            raise ValueError(f"Unknown retrieval method: {method}")

        bm25_order, bm25_scores = self._bm25_order(query)
        if method == "bm25":
            order = bm25_order
            scores = bm25_scores
        elif method == "dense":
            order, scores = self._dense_order(query)
        else:
            dense_order, dense_scores = self._dense_order(query)
            fused: dict[int, float] = {}
            for ranking in (bm25_order, dense_order):
                for position, index in enumerate(ranking, start=1):
                    fused[index] = fused.get(index, 0.0) + 1 / (60 + position)
            order = sorted(fused, key=lambda index: (-fused[index], index))
            scores = [fused[index] for index in range(len(self.tools))]
            if method == "rrf_rerank":
                shortlist = order[: min(8, len(order))]
                rerank_scores = list(self._load_reranker().rerank(
                    query,
                    [self.documents[index] for index in shortlist],
                ))
                if len(rerank_scores) != len(shortlist):
                    raise ValueError("Reranker must return one score for each candidate")
                scores_by_index = {
                    index: float(score) for index, score in zip(shortlist, rerank_scores)
                }
                order = sorted(
                    shortlist,
                    key=lambda index: (-scores_by_index[index], index),
                )
                for index, score in scores_by_index.items():
                    scores[index] = score

        return [
            {
                **self.tools[index],
                "score": float(scores[index]),
                "rank": position,
            }
            for position, index in enumerate(order[:top_k], start=1)
        ]

