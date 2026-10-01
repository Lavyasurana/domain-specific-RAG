"""Run with: LEGAL_RAG_PASSAGES=data/processed/passages.jsonl streamlit run app/streamlit_app.py"""

import os

import streamlit as st

from legal_rag.generation import generate_openai_answer
from legal_rag.hybrid import HybridRetriever
from legal_rag.io import load_passages

st.set_page_config(page_title="Evaluated Legal RAG")
st.title("Evaluated Legal RAG")
st.caption("Research assistance only — not legal advice. Verify every cited source.")
source = os.environ.get("LEGAL_RAG_PASSAGES")
if not source:
    st.info("Set LEGAL_RAG_PASSAGES to a processed passage JSONL file.")
    st.stop()
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Set OPENAI_API_KEY before starting the app. GPT-4o mini is required for generation.")
    st.stop()
@st.cache_resource
def get_retriever(path: str):
    return HybridRetriever(load_passages(path))
question = st.text_area("Legal research question")
if st.button("Search") and question.strip():
    context = get_retriever(source).search(question, 5)
    answer = generate_openai_answer(question, context, model="gpt-4o-mini")
    st.write(answer.answer)
    with st.expander("Retrieved evidence"):
        for item in context:
            st.markdown(f"**{item.passage.chunk_id}** · score {item.score:.3f}")
            st.write(item.passage.text)
