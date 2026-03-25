"""WhiteRAG Streamlit UI — Phase 6.

A clean, single-page interface for querying Ellen G. White's writings
with AI-generated answers and explicit citations.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.embeddings.store import DEFAULT_COLLECTION, DEFAULT_DB_PATH, DEFAULT_MODEL

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WhiteRAG",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 860px;
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2440 100%);
        border-left: 4px solid #4a9eff;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
        color: #e8f0fe;
        line-height: 1.75;
        font-size: 1.05rem;
    }

    /* Citation card */
    .citation-card {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
    }

    .citation-rank {
        display: inline-block;
        background: #4a9eff;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        text-align: center;
        line-height: 24px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
    }

    .citation-title {
        font-weight: 600;
        color: #93c5fd;
        font-size: 0.95rem;
    }

    .citation-meta {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .citation-text {
        color: #94a3b8;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        line-height: 1.6;
        border-top: 1px solid #1e3a5f;
        padding-top: 0.5rem;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        border-radius: 99px;
        padding: 1px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }

    .app-header h1 {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4a9eff, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .app-header p {
        color: #64748b;
        font-size: 1rem;
    }

    /* No results */
    .no-results {
        text-align: center;
        color: #64748b;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required for answer generation. Get one at platform.openai.com",
    )

    st.divider()

    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="OpenAI model used for generation",
    )

    top_k = st.slider(
        "Sources to retrieve",
        min_value=1, max_value=10, value=5,
        help="Number of text chunks retrieved from the vector store",
    )

    st.divider()

    show_excerpts = st.toggle("Show source excerpts", value=True)
    exclude_front_matter = st.toggle(
        "Exclude front matter",
        value=True,
        help="Filter out Preface, Foreword, Table of Contents, and other structural sections",
    )

    st.divider()
    st.markdown(
        "<small style='color:#475569'>📚 WhiteRAG — Citation-based RAG for "
        "Ellen G. White Writings</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cached resources — load once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading retriever…")
def get_retriever() -> Retriever:
    """Load and cache the Retriever (loads embedding model once)."""
    return Retriever(
        db_path=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION,
        model_name=DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>📖 WhiteRAG</h1>
    <p>Ask questions about Ellen G. White's writings — answers grounded in the source text.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder="e.g. What does Ellen White say about prayer and faith?",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run pipeline on submit
# ---------------------------------------------------------------------------

if submit and query.strip():

    retriever = get_retriever()

    # Check vector store
    info = retriever.collection_info()
    if info["count"] == 0:
        st.error(
            "⚠️ The vector store is empty. Run Phases 1–3 first:\n\n"
            "```bash\npython -m src.ingestion.cli\n"
            "python -m src.preprocessing.cli\n"
            "python -m src.embeddings.cli\n```"
        )
        st.stop()

    # Retrieve
    with st.spinner(f"Searching {info['count']:,} indexed passages…"):
        results = retriever.search(
            query.strip(),
            top_k=top_k,
            exclude_front_matter=exclude_front_matter,
        )

    if not results:
        st.markdown('<div class="no-results">No relevant passages found.</div>', unsafe_allow_html=True)
        st.stop()

    # Generate
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        generator = Generator(model=model, top_k=top_k)
        with st.spinner(f"Generating answer with {model}…"):
            try:
                response = generator.generate(query.strip(), results)
                st.markdown(
                    f'<div class="answer-card">{response.answer}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Generation error: {e}")
    else:
        st.warning("No OpenAI API key — showing retrieved sources only. Add your key in the sidebar to enable answer generation.")

    # Sources
    st.markdown(f"### 📚 Sources ({len(results)} retrieved)")

    for r in results:
        start, end = r.paragraph_range
        para = f"¶{start}" if start == end else f"¶{start}–{end}"
        citation = f"{r.book_title} — {r.chapter_title} [{para}]"
        preview = r.text[:350].rsplit(" ", 1)[0] + "…" if len(r.text) > 350 else r.text

        st.markdown(f"""
        <div class="citation-card">
            <div>
                <span class="citation-rank">{r.rank}</span>
                <span class="citation-title">{citation}</span>
                <span class="score-badge">{r.score:.2f}</span>
            </div>
            <div class="citation-meta">{r.word_count} words</div>
            {"<div class='citation-text'>" + preview + "</div>" if show_excerpts else ""}
        </div>
        """, unsafe_allow_html=True)

elif submit and not query.strip():
    st.warning("Please enter a question.")
