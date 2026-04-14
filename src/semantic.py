"""
semantic.py
-----------
Semantic (dense embedding) retrieval using sentence-transformers + FAISS.

Follows the approach specified in DSCI 575 Milestone 1, Step 3:

    Step 1 — Embeddings:
        Use sentence-transformers (all-MiniLM-L6-v2) to encode documents.

    Step 2 — Indexing:
        Use FAISS IndexFlatIP (inner product) with L2-normalised vectors.
        Since ||v|| = 1 after normalisation, inner product = cosine similarity.

    Persist with:
        faiss.write_index(index, path)
        faiss.read_index(path)

Usage
-----
    from src.semantic import SemanticRetriever
    from src.utils import build_corpus, build_metadata_lookup, load_jsonl_gz

    # Build
    reviews  = load_jsonl_gz('data/raw/Grocery_and_Gourmet_Food.jsonl.gz')
    metadata = load_jsonl_gz('data/raw/meta_Grocery_and_Gourmet_Food.jsonl.gz')
    lookup   = build_metadata_lookup(metadata)
    docs     = build_corpus(reviews, lookup)

    retriever = SemanticRetriever()
    retriever.build_index(docs)
    retriever.save(
        index_path='data/processed/semantic.index',
        docs_path='data/processed/semantic_docs.pkl'
    )

    # Search
    results = retriever.search("something to keep coffee hot all morning", top_k=5)
    for r in results:
        print(r['rank'], r['title'], f"{r['score']:.4f}")

    # Reload
    retriever2 = SemanticRetriever()
    retriever2.load(
        index_path='data/processed/semantic.index',
        docs_path='data/processed/semantic_docs.pkl'
    )
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """
    Dense retrieval via sentence-transformer embeddings + FAISS IndexFlatIP.

    Encoding pipeline (matches Lecture 5):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(documents)   # shape (n_docs, 384)

    Index:
        faiss.normalize_L2(embeddings)         # unit-normalize
        index = faiss.IndexFlatIP(384)         # inner product = cosine sim
        index.add(embeddings)

    At query time:
        query_emb = model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, top_k)

    Attributes
    ----------
    model     : SentenceTransformer instance
    index     : faiss.IndexFlatIP
    documents : list of document dicts (parallel to FAISS index positions)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Parameters
        ----------
        model_name : HuggingFace model identifier.
                     Defaults to all-MiniLM-L6-v2, the same model used in
                     DSCI 563 Lab 1 and referenced in Lecture 5.
        """
        print(f"Loading sentence-transformer model: {model_name} ...")
        self.model     = SentenceTransformer(model_name)
        self.index     = None
        self.documents = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, documents: list[dict], batch_size: int = 64) -> None:
        """
        Encode all documents and build a FAISS IndexFlatIP.

        Uses `combined_text` from each document — the same field as BM25 —
        so both retrievers operate over the same textual representation.

        Why IndexFlatIP + L2 normalisation?
            After normalising to unit length, the inner product between two
            vectors equals their cosine similarity. This is the same metric
            as sklearn.metrics.pairwise.cosine_similarity but served through
            FAISS for scalable approximate or exact nearest-neighbour search.

        Parameters
        ----------
        documents  : list of dicts from utils.build_corpus()
        batch_size : encoding batch size (reduce if out of memory)
        """
        self.documents = documents
        texts = [doc['combined_text'] for doc in documents]

        print(f"Encoding {len(texts)} documents with {self.model._modules['0'].auto_model.config.model_type} ...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,   # we normalise manually below
        ).astype('float32')

        # L2-normalise so that inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        dim         = embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"FAISS index ready — {self.index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top-k documents for a query using cosine similarity.

        Parameters
        ----------
        query  : raw user query string
        top_k  : number of results to return

        Returns
        -------
        list of document dicts, each augmented with:
            score : cosine similarity in [-1, 1] (higher = more similar)
            rank  : 1-based rank position (int)
        """
        if self.index is None:
            raise RuntimeError(
                "Index not built. Call build_index() or load() first."
            )

        # Encode and normalise the query
        query_emb = self.model.encode(
            [query], convert_to_numpy=True
        ).astype('float32')
        faiss.normalize_L2(query_emb)

        # FAISS search: scores shape (1, top_k), indices shape (1, top_k)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if idx == -1:       # FAISS returns -1 when fewer docs than top_k
                break
            doc          = self.documents[idx].copy()
            doc['score'] = float(score)
            doc['rank']  = rank
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, docs_path: str) -> None:
        """
        Persist the FAISS index and document store to disk.

        Uses faiss.write_index() as specified in Milestone 1 Step 3.

        Parameters
        ----------
        index_path : e.g. 'data/processed/semantic.index'
        docs_path  : e.g. 'data/processed/semantic_docs.pkl'
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"FAISS index saved  → {index_path}")
        print(f"Documents saved    → {docs_path}")

    def load(self, index_path: str, docs_path: str) -> None:
        """
        Load a previously saved FAISS index and document store.

        Uses faiss.read_index() as specified in Milestone 1 Step 3.

        Parameters
        ----------
        index_path : path written by save()
        docs_path  : path written by save()
        """
        self.index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"FAISS index loaded ← {index_path} ({self.index.ntotal} vectors)")
        print(f"Documents loaded   ← {docs_path} ({len(self.documents)} docs)")
