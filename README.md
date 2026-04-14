# DSCI 575 — Information Retrieval with BM25 and Embeddings
**Team**: `Roganci Fontelera`  
**Dataset**: Amazon Reviews 2023 — Grocery and Gourmet Food category

---

## Project Description

This project implements and compares two information retrieval systems over Amazon book reviews:

1. **BM25** — classic keyword-based retrieval (Okapi BM25 via `rank_bm25`)
2. **Semantic Search** — dense embedding retrieval (`all-MiniLM-L6-v2` + FAISS)

A Streamlit web app lets users enter a query and toggle between retrieval methods.

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── .env                         # API keys — never committed
├── data/
│   ├── raw/                     # .jsonl.gz files (git-ignored)
│   └── processed/               # built indexes (git-ignored)
├── notebooks/
│   └── milestone1_exploration.ipynb
├── src/
│   ├── utils.py                 # shared data loading & tokenisation
│   ├── bm25.py                  # BM25Retriever class
│   └── semantic.py              # SemanticRetriever class
├── results/
│   └── milestone1_discussion.md
└── app/
    └── app.py                   # Streamlit app
```

---

## Dataset

- **Source**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- **Category used**: Grocery and Gourmet Food
- **Review file**: `Grocery_and_Gourmet_Food.jsonl` — user-written reviews, ratings, timestamps
- **Metadata file**: `meta_Grocery_and_Gourmet_Food.jsonl` — product titles, descriptions, features, price

Download both files and place them in `data/raw/`. Do **not** commit them to GitHub (they are in `.gitignore`).

### Fields used for retrieval

| File     | Field         | Reason |
|----------|---------------|--------|
| Metadata | `title`       | Product name — brand, type, flavour keywords |
| Metadata | `description` | Manufacturer description — ingredients, dietary info |
| Metadata | `features`    | Size, certifications (organic, gluten-free, vegan) |
| Review   | `title`       | Punchy user headline |
| Review   | `text`        | Full review body — taste notes, comparisons, use cases |

All five fields are concatenated into a single `combined_text` per document.

---

## Environment Setup

```bash
# 1. Clone the repo
git clone git@github.com:UBC-MDS/DSCI_575_project_<cwl1>_<cwl2>.git
cd DSCI_575_project_<cwl1>_<cwl2>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file (if using API keys in later milestones)
cp .env.example .env
# Edit .env with your keys
```

### Conda alternative

```bash
conda create -n dsci575 python=3.10
conda activate dsci575
pip install -r requirements.txt
```

---

## Retrieval Workflows

### BM25

1. Load reviews + metadata → merge into corpus via `utils.build_corpus()`
2. Tokenise each document's `combined_text` (lowercase, remove punctuation + stopwords)
3. Build `BM25Okapi` index over tokenised corpus
4. At query time, tokenise query identically, call `bm25.get_scores()`
5. Persist index with `pickle` to `data/processed/bm25_index.pkl`

### Semantic Search

1. Same corpus as BM25
2. Encode each document with `all-MiniLM-L6-v2` (384-dim embeddings, L2-normalised)
3. Build `faiss.IndexFlatIP` (inner product = cosine similarity on normalised vectors)
4. At query time, encode query with same model, call `index.search()`
5. Persist with `faiss.write_index` + `pickle`

---

## Running the App Locally

```bash
# Build indexes first (only needed once)
# Option A: via the notebook
jupyter notebook notebooks/milestone1_exploration.ipynb
# Run cells 10–12 to build and save both indexes

# Option B: let the app build them for you
# Just launch the app and click "Build Indexes" in the UI

# Launch app
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Select a retrieval method from the sidebar and enter your query.

---

## Reproducing the EDA

```bash
jupyter notebook notebooks/milestone1_exploration.ipynb
```

Run all cells top-to-bottom. The first 200 records are used for EDA; cells 10–12 build full indexes.
