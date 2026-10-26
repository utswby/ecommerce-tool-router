# Lightweight E-commerce Tool-Routing Project

This standalone project narrows the original guide agent to one NLP question: can a user request be matched to the right e-commerce tool descriptions? It compares lexical retrieval, pretrained semantic embeddings, and rank fusion on a small synthetic benchmark. A Streamlit demo shows the selected tools and their simulated results.

The project uses English queries and eight representative tools. It does not fine-tune a language model, call an external API, or require a GPU. The dense and reranking models are downloaded only when those methods are selected; FastEmbed runs them on CPU and keeps model files in its normal local cache, outside this project folder.

## Methods

- `bm25`: lexical retrieval using BM25.
- `dense`: cosine similarity over a pretrained BGE-small sentence embedding model.
- `rrf`: reciprocal-rank fusion of BM25 and dense retrieval.
- `rrf_rerank`: rerank the hybrid shortlist with a small pretrained cross-encoder. This is optional and downloads an additional model.

The benchmark contains held-out synthetic queries from the original project's test split, filtered to this project's eight tools, plus a small hand-authored set of paraphrases and multi-intent queries. Synthetic data is useful for a controlled comparison, but results should not be presented as evidence of real customer performance.

## Setup

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Install the optional CPU embedding and reranking runtime only when running the full comparison:

```powershell
python -m pip install -r requirements-embeddings.txt
```

Start the BM25 demo without installing or downloading any model:

```powershell
streamlit run app.py
```

BM25 is selected by default. Select `Dense`, `Hybrid RRF`, or `Hybrid + reranker` in the sidebar to download and load the corresponding model on first use. The embedding model is `BAAI/bge-small-en-v1.5`; the optional reranker is `Xenova/ms-marco-MiniLM-L-6-v2`.

## Run the benchmark

Run the zero-download lexical baseline first:

```powershell
python evaluate.py --methods bm25
```

Run the full comparison after installing `requirements-embeddings.txt` and allowing the pretrained model downloads:

```powershell
python evaluate.py --methods bm25,dense,rrf,rrf_rerank
```

The output is saved to `results/evaluation_results.json`. `--limit 20` is useful for a quick smoke test. The benchmark reports exact tool-set accuracy, micro and macro F1, Recall@1/3, and warmed per-query latency. Exact set accuracy selects as many top-ranked tools as appear in that query's known label; this isolates ranking quality from a separate decision-threshold problem. State that choice when interpreting results.

The initial BM25 reference run on all 96 examples is saved in `results/bm25_baseline.json` (exact tool-set accuracy 0.719, micro-F1 0.775, Recall@3 0.901). Regenerate it with the BM25 command above before using the numbers in a report.

All methods use the same fixed evaluation set. No labels are used to fit the embedding model or reranker. The test records are synthetic and the hand-authored queries are kept in a separate file so their contribution can be reported clearly.

## Project layout

```text
app.py                       Streamlit demo
evaluate.py                  Reproducible method comparison
commerce_lite/               Retrieval, catalog loading, and metrics
data/tools.json              Eight tool descriptions and mock responses
data/evaluation_queries.json Filtered held-out synthetic test records
data/manual_queries.json     Hand-authored single- and multi-intent examples
tests/                       Fast tests that do not download models
```

The original project remains separate and is not needed to run this folder.
