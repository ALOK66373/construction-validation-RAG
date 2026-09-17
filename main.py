"""Streamlit app for the Construction Building Code Advisor."""

import os
import sys
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RAGEngine


rag_engine: Optional[RAGEngine] = None


@st.cache_resource
def get_rag_engine() -> RAGEngine:
    global rag_engine
    rag_engine = RAGEngine(data_dir="./data", chroma_dir="./chroma_db")
    return rag_engine


def render_app() -> None:
    st.set_page_config(page_title="Building Code Advisor", page_icon="🏗️", layout="wide")
    st.title("🏗️ Construction Building Code Advisor")
    st.caption("Grounded answers from Bangalore bylaws and building code documents via Groq")

    engine = get_rag_engine()

    with st.sidebar:
        st.header("Controls")
        n_results = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)
        st.write(f"Knowledge base loaded: {not engine.vector_store.is_empty()}")
        st.write(f"Total chunks: {engine.vector_store.count()}")

    query = st.text_area(
        "Ask a building code question",
        value="What approvals are needed for a second floor in Bangalore?",
        height=120,
    )

    if st.button("Ask the advisor") and query.strip():
        with st.spinner("Searching the knowledge base and generating an answer..."):
            result = engine.answer(query=query, n_results=n_results)

        st.subheader("Answer")
        answer_text = result.get("answer", "").strip()
        if answer_text:
            st.markdown(answer_text)
        else:
            st.info("No answer was generated for this question.")

        st.subheader("Source documents")
        if result["sources"]:
            for index, source in enumerate(result["sources"], start=1):
                with st.expander(
                    f"{source['source']} — chunk {source['chunk_index']} (relevance {source['relevance_percent']}%)",
                    expanded=index == 1,
                ):
                    st.progress(source["relevance_percent"] / 100)
                    st.write(source["preview"])
        else:
            st.info("No source documents were returned for this query.")


def run_streamlit_app() -> None:
    render_app()


if __name__ == "__main__":
    run_streamlit_app()

