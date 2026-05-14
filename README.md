---
title: MovieMate
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.31.0"
app_file: app.py
pinned: false
license: mit
---

# 🎬 MovieMate — Conversational AI Movie Assistant

> Exploring Conversational AI for Intelligent Movie Search and Recommendations  
> NLP Assignment — Hugging Face Spaces Deployment

---

## What it does

MovieMate lets you discover movies through natural conversation instead of rigid keyword search.  
Ask in plain English — it retrieves semantically similar films and explains its recommendations.

**Example queries**
- *"Recommend a sci-fi movie like Inception"*
- *"Movies starring Leonardo DiCaprio after 2010"*
- *"Best feel-good films for a Friday night"*
- *"Who directed Interstellar?"*

---

## Architecture

| Layer | Technology |
|---|---|
| Dataset | TMDB API — ~1 000 movies with genres, cast, director |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector search | FAISS `IndexFlatL2` |
| Response generation | Gemini 1.5 Flash |
| UI | Gradio 4 `Blocks` |

**Pipeline per query:**
1. Encode user query → 384-dim embedding  
2. FAISS nearest-neighbour search (k=3)  
3. Format retrieved movies + conversation history as context  
4. Gemini generates a conversational, explained recommendation

---

## Setup (first-time / local)

### 1. Clone and install
```bash
git clone https://huggingface.co/spaces/<your-username>/moviemate
cd moviemate
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export GEMINI_API_KEY="your-gemini-key"   # from https://aistudio.google.com
export TMDB_API_KEY="your-tmdb-key"       # from https://www.themoviedb.org/settings/api
```

### 3. (Optional) Pre-fetch the dataset
If `movies_updated.csv` is absent the app fetches it automatically on first launch (takes ~5 min).  
To pre-generate and commit it:
```bash
python scripts/fetch_dataset.py
```

### 4. Run locally
```bash
python app.py
```

---

## Hugging Face Spaces deployment

### Secrets (required)
In your Space → **Settings → Repository secrets**, add:

| Secret name | Value |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `TMDB_API_KEY` | Your TMDB API key (only needed if CSV is absent) |

### Committing the dataset (recommended)
Committing `movies_updated.csv` to the repo avoids the TMDB fetch on every cold start:
```bash
git lfs install           # HF uses LFS for large files automatically
git add movies_updated.csv
git commit -m "add pre-fetched dataset"
git push
```

---

## File structure

```
moviemate-hf/
├── app.py                  # Main application (Gradio + RAG pipeline)
├── requirements.txt        # Python dependencies
├── README.md               # This file (also the HF Space card)
├── scripts/
│   └── fetch_dataset.py    # Standalone script to pre-build movies_updated.csv
└── movies_updated.csv      # Pre-fetched dataset (commit this to avoid cold-start delay)
```

---

## Notes

- **Cold start**: If `movies_updated.csv` is absent, the app fetches ~1 000 movies from TMDB (≈5 min). Commit the CSV to avoid this.  
- **Gemini quota**: The free tier allows ~60 requests/min. The app retries up to 3 times with back-off.  
- **FAISS on CPU**: `faiss-cpu` is used for HF compatibility. Search over 1 000 movies is near-instant.
