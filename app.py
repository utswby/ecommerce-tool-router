from __future__ import annotations

import streamlit as st

from commerce_lite.catalog import load_tools
from commerce_lite.retrieval import RetrievalEngine


@st.cache_resource
def get_engine() -> RetrievalEngine:
    return RetrievalEngine(load_tools())


METHODS = {
    "BM25": "bm25",
    "Dense embeddings": "dense",
    "Hybrid RRF": "rrf",
    "Hybrid + reranker": "rrf_rerank",
}


def main() -> None:
    st.set_page_config(page_title="E-commerce Tool Router", layout="wide")
    st.title("E-commerce Tool Router")

    with st.sidebar:
        selected_method = st.selectbox("Retrieval method", list(METHODS))
        tool_count = st.slider("Tools to return", min_value=1, max_value=3, value=1)
        st.caption("Semantic models are fetched the first time a model-based method is used.")

    st.caption("Enter a shopping or after-sales request to see the matching tools.")
    query = st.chat_input("For example: check the price and stock of SKU-001")
    if not query:
        return

    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        try:
            ranked = get_engine().rank(query, method=METHODS[selected_method], top_k=tool_count)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        if not ranked:
            st.info("No matching tool was found.")
            return

        st.write("Selected simulated service results:")
        for item in ranked:
            st.markdown(f"**{item['name']}** · {item['service']}")
            st.write(item["demo_result"])

        with st.expander("Ranking details"):
            st.dataframe(
                [
                    {"Rank": item["rank"], "Tool": item["name"], "Score": round(item["score"], 6)}
                    for item in ranked
                ],
                hide_index=True,
            )


if __name__ == "__main__":
    main()
