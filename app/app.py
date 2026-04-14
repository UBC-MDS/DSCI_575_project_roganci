"""
app.py
------
Streamlit web application for BM25 and Semantic retrieval over
Amazon Grocery and Gourmet Food reviews.

Run:
    streamlit run app/app.py

Expects pre-built indexes at:
    data/processed/bm25_index.pkl
    data/processed/semantic.index
    data/processed/semantic_docs.pkl

If indexes are missing, the app offers to build them from raw data files.
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — works whether launched from repo root or from app/ directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever
from src.utils import build_corpus, build_metadata_lookup, load_jsonl_gz

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BM25_INDEX_PATH    = ROOT / 'data/processed/bm25_index.pkl'
SEMANTIC_INDEX_PATH = str(ROOT / 'data/processed/semantic.index')
SEMANTIC_DOCS_PATH  = str(ROOT / 'data/processed/semantic_docs.pkl')
REVIEWS_PATH        = ROOT / 'data/raw/Grocery_and_Gourmet_Food.jsonl.gz'
META_PATH           = ROOT / 'data/raw/meta_Grocery_and_Gourmet_Food.jsonl.gz'

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Grocery Search",
    page_icon="🛒",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #4a90d9;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .result-card h4 { margin: 0 0 4px 0; color: #1a1a2e; }
    .result-meta { color: #666; font-size: 0.85em; margin-bottom: 6px; }
    .review-text { font-size: 0.92em; color: #333; line-height: 1.5; }
    .score-badge {
        display: inline-block;
        background: #4a90d9;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 8px;
    }
    .no-results { color: #999; text-align: center; padding: 40px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BM25 index ...")
def load_bm25() -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.load(str(BM25_INDEX_PATH))
    return retriever


@st.cache_resource(show_spinner="Loading semantic index ...")
def load_semantic() -> SemanticRetriever:
    retriever = SemanticRetriever()
    retriever.load(SEMANTIC_INDEX_PATH, SEMANTIC_DOCS_PATH)
    return retriever


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion  (Lecture 5)
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    bm25_results: list[dict],
    sem_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion (RRF).

    From Lecture 5:
        RRF(d) = sum over retrieval methods r of  1 / (k + rank_r(d))

    where k = 60 is the standard smoothing constant.

    Each document is identified by (asin, review_title) so that the same
    product review appearing in both lists is correctly merged.

    Parameters
    ----------
    bm25_results : ranked list from BM25Retriever.search()
    sem_results  : ranked list from SemanticRetriever.search()
    k            : RRF smoothing constant (default 60 per lecture)
    top_k        : number of final results to return

    Returns
    -------
    list of merged document dicts sorted by descending RRF score,
    each augmented with:
        score : RRF score (float)
        rank  : 1-based final rank (int)
    """
    rrf_scores: dict[str, dict] = {}

    def doc_key(doc: dict) -> str:
        return doc.get('asin', '') + '||' + doc.get('review_title', '')

    # Accumulate RRF contributions from BM25
    for doc in bm25_results:
        key = doc_key(doc)
        if key not in rrf_scores:
            rrf_scores[key] = {'doc': doc, 'rrf': 0.0}
        rrf_scores[key]['rrf'] += 1.0 / (k + doc['rank'])

    # Accumulate RRF contributions from semantic
    for doc in sem_results:
        key = doc_key(doc)
        if key not in rrf_scores:
            rrf_scores[key] = {'doc': doc, 'rrf': 0.0}
        rrf_scores[key]['rrf'] += 1.0 / (k + doc['rank'])

    # Sort by RRF score descending
    merged = sorted(rrf_scores.values(), key=lambda x: x['rrf'], reverse=True)

    results = []
    for rank, entry in enumerate(merged[:top_k], start=1):
        doc          = entry['doc'].copy()
        doc['score'] = entry['rrf']
        doc['rank']  = rank
        results.append(doc)

    return results


# ---------------------------------------------------------------------------
# Index builder UI (shown when processed files are missing)
# ---------------------------------------------------------------------------
def build_indexes_ui():
    st.warning(
        "Pre-built indexes not found in `data/processed/`. "
        "Build them now from your raw data files."
    )
    max_docs = st.number_input(
        "Max documents to index (e.g. 5 000 for a quick test)",
        min_value=100, max_value=500_000, value=5000, step=500,
    )
    if st.button("Build Indexes"):
        if not REVIEWS_PATH.exists() or not META_PATH.exists():
            st.error(
                f"Raw data files not found:\n"
                f"  {REVIEWS_PATH}\n  {META_PATH}\n\n"
                "Download them from https://amazon-reviews-2023.github.io/ "
                "and place them in `data/raw/`."
            )
            return

        with st.spinner("Loading raw data ..."):
            reviews  = load_jsonl_gz(str(REVIEWS_PATH), max_records=int(max_docs))
            metadata = load_jsonl_gz(str(META_PATH))
            lookup   = build_metadata_lookup(metadata)
            docs     = build_corpus(reviews, lookup)
        st.info(f"Corpus built: {len(docs)} documents.")

        with st.spinner("Building BM25 index ..."):
            bm25 = BM25Retriever()
            bm25.build_index(docs)
            bm25.save(str(BM25_INDEX_PATH))

        with st.spinner("Building semantic index (this may take a few minutes) ..."):
            sem = SemanticRetriever()
            sem.build_index(docs)
            sem.save(SEMANTIC_INDEX_PATH, SEMANTIC_DOCS_PATH)

        st.success("Indexes built! Reload the page to start searching.")
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stars(rating) -> str:
    if rating is None:
        return "No rating"
    try:
        n = round(float(rating))
        return "⭐" * n + "☆" * max(0, 5 - n) + f"  ({float(rating):.1f})"
    except (ValueError, TypeError):
        return str(rating)


def render_result(doc: dict, rank: int) -> None:
    title       = doc.get('title') or "*(no title)*"
    review_text = doc.get('review_text') or ""
    rating      = doc.get('rating')
    score       = doc.get('score', 0.0)

    st.markdown(f"""
    <div class="result-card">
        <h4>#{rank} — {title}
            <span class="score-badge">score: {score:.4f}</span>
        </h4>
        <div class="result-meta">{stars(rating)}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Read review"):
        st.write(review_text)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Amazon Grocery & Gourmet Food — Retrieval Search")
st.caption(
    "Compare BM25 keyword search vs. semantic embedding search "
    "(all-MiniLM-L6-v2 + FAISS). Hybrid uses Reciprocal Rank Fusion."
)

# Check indexes exist
bm25_ready     = BM25_INDEX_PATH.exists()
semantic_ready = (
    Path(SEMANTIC_INDEX_PATH).exists()
    and Path(SEMANTIC_DOCS_PATH).exists()
)

if not bm25_ready or not semantic_ready:
    build_indexes_ui()
    st.stop()

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Search Settings")

    method = st.radio(
        "Retrieval method",
        options=["BM25", "Semantic", "Hybrid (RRF)"],
        index=0,
    )
    top_k = st.slider("Results to show", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("- **BM25** — works best with exact keywords")
    st.markdown("- **Semantic** — describe concepts or intent in natural language")
    st.markdown("- **Hybrid (RRF)** — combines both via Reciprocal Rank Fusion")

# --- Query input -----------------------------------------------------------
query = st.text_input(
    "Enter your search query",
    placeholder="e.g. organic olive oil, coffee that is smooth and not bitter ...",
)
search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    bm25_r = load_bm25()
    sem_r  = load_semantic()

    # ---- BM25 only ---------------------------------------------------------
    if method == "BM25":
        with st.spinner("Searching ..."):
            results = bm25_r.search(query, top_k=top_k)
        st.subheader(f"BM25 results for: *{query}*")
        for r in results:
            render_result(r, r['rank'])
        if not results:
            st.markdown('<div class="no-results">No results found.</div>',
                        unsafe_allow_html=True)

    # ---- Semantic only -----------------------------------------------------
    elif method == "Semantic":
        with st.spinner("Encoding query and searching ..."):
            results = sem_r.search(query, top_k=top_k)
        st.subheader(f"Semantic results for: *{query}*")
        for r in results:
            render_result(r, r['rank'])
        if not results:
            st.markdown('<div class="no-results">No results found.</div>',
                        unsafe_allow_html=True)

    # ---- Hybrid (RRF) ------------------------------------------------------
    else:
        # Retrieve a larger candidate pool from each method before fusion
        candidate_k = top_k * 3
        with st.spinner("Running hybrid search (RRF) ..."):
            bm25_results = bm25_r.search(query, top_k=candidate_k)
            sem_results  = sem_r.search(query, top_k=candidate_k)
            hybrid_results = reciprocal_rank_fusion(
                bm25_results, sem_results, k=60, top_k=top_k
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("BM25 (input)")
            for r in bm25_results[:top_k]:
                render_result(r, r['rank'])
        with col2:
            st.subheader("Semantic (input)")
            for r in sem_results[:top_k]:
                render_result(r, r['rank'])

        st.subheader(f"Hybrid — RRF fusion (k=60)")
        st.caption(
            "RRF score = 1/(60 + rank_BM25) + 1/(60 + rank_Semantic) "
            "for each document. No score normalisation needed."
        )
        for r in hybrid_results:
            render_result(r, r['rank'])

elif search_clicked:
    st.warning("Please enter a query before searching.")
