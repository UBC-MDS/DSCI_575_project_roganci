"""
bm25.py
-------
BM25 keyword-based retrieval using the rank_bm25 library.

Follows the pattern demonstrated in DSCI 575 Lecture 5:

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [simple_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

Usage
-----
    from src.bm25 import BM25Retriever
    from src.utils import build_corpus, build_metadata_lookup, load_jsonl_gz

    # Build index
    reviews  = load_jsonl_gz('data/raw/Grocery_and_Gourmet_Food.jsonl.gz')
    metadata = load_jsonl_gz('data/raw/meta_Grocery_and_Gourmet_Food.jsonl.gz')
    lookup   = build_metadata_lookup(metadata)
    docs     = build_corpus(reviews, lookup)

    retriever = BM25Retriever()
    retriever.build_index(docs)
    retriever.save('data/processed/bm25_index.pkl')

    # Search
    results = retriever.search("organic olive oil cold pressed", top_k=5)
    for r in results:
        print(r['rank'], r['title'], f"{r['score']:.3f}")

    # Reload
    retriever2 = BM25Retriever()
    retriever2.load('data/processed/bm25_index.pkl')
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi with build / search / persist helpers.

    The tokenizer applied to the corpus and to every query is the shared
    `tokenize()` function from utils.py (lowercase → remove punctuation →
    split → optionally remove stopwords). Using the same tokenizer at index
    time and query time is critical for BM25 term-frequency statistics to
    be meaningful.

    Attributes
    ----------
    bm25             : BM25Okapi instance
    documents        : list of document dicts (original, un-tokenised)
    tokenized_corpus : list of token lists used to build the index
    """

    def __init__(self):
        self.bm25             = None
        self.documents        = None
        self.tokenized_corpus = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, documents: list[dict]) -> None:
        """
        Tokenise all documents and build the BM25 index.

        Each document's `combined_text` field is tokenised with the shared
        `tokenize()` function. The same tokenizer is used at query time.

        Parameters
        ----------
        documents : list of dicts from `utils.build_corpus()`
        """
        self.documents = documents

        print(f"Tokenising {len(documents)} documents ...")
        self.tokenized_corpus = [
            tokenize(doc['combined_text']) for doc in documents
        ]

        print("Building BM25Okapi index ...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index ready — {len(documents)} documents indexed.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top-k documents for a query.

        Follows the Lecture 5 pattern:
            tokenized_query = simple_tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            ranked_idx = sorted(..., key=lambda i: scores[i], reverse=True)

        Parameters
        ----------
        query  : raw user query string
        top_k  : number of results to return

        Returns
        -------
        list of document dicts, each augmented with:
            score : BM25 relevance score (float, higher = more relevant)
            rank  : 1-based rank position (int)
        """
        if self.bm25 is None:
            raise RuntimeError(
                "Index not built. Call build_index() or load() first."
            )

        tokenized_query = tokenize(query)
        scores          = self.bm25.get_scores(tokenized_query)

        # Sort descending by score, take top_k
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_idx, start=1):
            doc          = self.documents[idx].copy()
            doc['score'] = float(scores[idx])
            doc['rank']  = rank
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the index and document store to a pickle file.

        Saves the BM25Okapi object, the original documents, and the
        tokenized corpus together so the index can be fully restored.

        Parameters
        ----------
        path : destination file path (e.g. 'data/processed/bm25_index.pkl')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'bm25':             self.bm25,
            'documents':        self.documents,
            'tokenized_corpus': self.tokenized_corpus,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"BM25 index saved → {path}")

    def load(self, path: str) -> None:
        """
        Deserialise a previously saved index.

        Parameters
        ----------
        path : source file path written by save()
        """
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self.bm25             = payload['bm25']
        self.documents        = payload['documents']
        self.tokenized_corpus = payload['tokenized_corpus']
        print(f"BM25 index loaded ← {path} ({len(self.documents)} docs)")
