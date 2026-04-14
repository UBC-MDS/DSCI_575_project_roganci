"""
utils.py
--------
Shared utilities for corpus construction, tokenization, and data loading.
Used by both bm25.py and semantic.py.

Tokenization follows the approach demonstrated in DSCI 575 Lecture 5:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    tokens = text.split()
with optional stopword removal on top.
"""

import gzip
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Stopword list — no NLTK dependency required
# ---------------------------------------------------------------------------
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
    "won't", 'wouldn', "wouldn't",
}


def load_jsonl_gz(filepath: str, max_records: int = None) -> list[dict]:
    """
    Load records from a .jsonl.gz file line by line.

    Parameters
    ----------
    filepath    : path to .jsonl.gz file
    max_records : if set, stop after this many records (useful for EDA)

    Returns
    -------
    list of dicts, one per record
    """
    records = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Normalise and tokenise a string for BM25.

    Follows the approach from DSCI 575 Lecture 5 (simple_tokenize):
        1. Lowercase
        2. Remove non-alphanumeric characters (keep hyphens)
        3. Split on whitespace
        4. (Optionally) remove stopwords

    Parameters
    ----------
    text             : raw input string
    remove_stopwords : whether to filter stopwords

    Returns
    -------
    list of token strings
    """
    if not text or not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens


def build_metadata_lookup(metadata: list[dict]) -> dict:
    """
    Convert a flat list of metadata dicts into a dict keyed by parent_asin.

    Parameters
    ----------
    metadata : list of metadata dicts loaded from meta_*.jsonl.gz

    Returns
    -------
    dict  {parent_asin: metadata_dict}
    """
    return {
        item['parent_asin']: item
        for item in metadata
        if 'parent_asin' in item
    }


def build_corpus(
    reviews: list[dict],
    metadata_lookup: dict,
    max_docs: int = None,
) -> list[dict]:
    """
    Merge review records with product metadata into unified retrieval documents.

    Fields selected for retrieval (Grocery and Gourmet Food category):
      - title        : product title — brand, flavour, format keywords
      - description  : manufacturer description — ingredients, dietary claims
      - features     : bullet-point attributes — size, certifications (organic,
                       gluten-free, vegan, kosher)
      - review_title : user-written headline — punchy taste summary
      - review_text  : full review body — taste notes, use cases, comparisons

    All five fields are concatenated into `combined_text` which is the single
    field indexed by BM25 and encoded by the sentence-transformer.

    Parameters
    ----------
    reviews          : list of review dicts from load_jsonl_gz()
    metadata_lookup  : dict from build_metadata_lookup()
    max_docs         : optional cap on number of output documents

    Returns
    -------
    list of document dicts, each containing:
        asin, title, review_title, review_text, rating,
        description, features, combined_text

    Notes
    -----
    Records with no product title are excluded. A missing title means the
    review has no matching metadata entry, so the document cannot be
    meaningfully presented to a user. This is a deliberate preprocessing
    decision documented in results/milestone1_discussion.md.
    """
    documents = []

    for i, review in enumerate(reviews):
        if max_docs is not None and i >= max_docs:
            break

        asin = review.get('asin', '')
        meta = metadata_lookup.get(asin, {})

        title        = (meta.get('title') or '').strip()
        description  = ' '.join(meta.get('description') or []).strip()
        features     = ' '.join(meta.get('features') or []).strip()
        review_title = (review.get('title') or '').strip()
        review_text  = (review.get('text') or '').strip()
        rating       = review.get('rating')

        combined_text = ' '.join(
            part for part in [title, description, features, review_title, review_text]
            if part
        )

        # Exclude records with no product title — a document that cannot
        # display a product name is not useful to a real user, and missing
        # titles indicate the review has no matching metadata entry.
        if not combined_text or not title:
            continue

        documents.append({
            'asin':          asin,
            'title':         title,
            'review_title':  review_title,
            'review_text':   review_text,
            'rating':        rating,
            'description':   description,
            'features':      features,
            'combined_text': combined_text,
        })

    return documents
