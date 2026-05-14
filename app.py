"""
MovieMate — Hugging Face Spaces
Conversational AI for intelligent movie search and recommendations.
"""

import os
import time
import numpy as np
import pandas as pd
import faiss
import gradio as gr
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────
# 1. API Configuration — keys come from HF Space secrets
# ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TMDB_API_KEY   = os.environ.get("TMDB_API_KEY", "")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set.\n"
        "Go to your HF Space → Settings → Repository secrets and add GEMINI_API_KEY."
    )

genai.configure(api_key=GEMINI_API_KEY)
model_llm = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = """
You are 'MovieMate', an intelligent and friendly movie assistant.
Use the retrieved movie details to answer the user's request.
Be conversational, helpful, and explain WHY you recommend each film.
If a movie isn't a perfect match, acknowledge it but still provide value.
Keep responses concise (2-4 sentences per recommendation).
"""

# ──────────────────────────────────────────────────────────────
# 2. Dataset loading
# ──────────────────────────────────────────────────────────────
DATA_FILE = "movies_updated.csv"


def fetch_from_tmdb() -> pd.DataFrame:
    """Fallback: fetch fresh data from TMDB if CSV is missing."""
    import requests

    if not TMDB_API_KEY:
        raise EnvironmentError(
            "movies_updated.csv not found AND TMDB_API_KEY is not set.\n"
            "Either commit movies_updated.csv to the Space or add TMDB_API_KEY secret."
        )

    print("Fetching movie list from TMDB (50 pages)...")
    frames = []
    for page in range(1, 51):
        try:
            r = requests.get(
                "https://api.themoviedb.org/3/discover/movie",
                params={"api_key": TMDB_API_KEY, "language": "en", "page": page},
                timeout=10,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                break
            frames.append(
                pd.DataFrame(results)[
                    ["id", "title", "overview", "release_date",
                     "popularity", "vote_average", "vote_count"]
                ]
            )
        except Exception as e:
            print(f"  Page {page} error: {e}")
        time.sleep(0.1)

    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df)} movies. Enriching with details...")

    ratings, genres_list, directors, casts, durations, years = [], [], [], [], [], []
    for movie_id in df["id"]:
        try:
            r = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": TMDB_API_KEY, "append_to_response": "credits"},
                timeout=10,
            )
            data = r.json()
            ratings.append(data.get("vote_average"))
            genres_list.append(", ".join(g["name"] for g in data.get("genres", [])))
            director = next(
                (c["name"] for c in data.get("credits", {}).get("crew", [])
                 if c["job"] == "Director"), None
            )
            directors.append(director)
            casts.append(
                ", ".join(c["name"] for c in data.get("credits", {}).get("cast", [])[:5])
            )
            durations.append(data.get("runtime"))
            rd = data.get("release_date", "")
            years.append(rd.split("-")[0] if rd else None)
        except Exception as e:
            print(f"  Movie {movie_id}: {e}")
            ratings.append(None); genres_list.append(None); directors.append(None)
            casts.append(None);   durations.append(None);  years.append(None)
        time.sleep(0.2)

    df["rating"]   = ratings
    df["genres"]   = genres_list
    df["director"] = directors
    df["cast"]     = casts
    df["duration"] = durations
    df["year"]     = years
    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset saved to {DATA_FILE}")
    return df


def load_dataframe() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        print(f"Loading dataset from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
    else:
        df = fetch_from_tmdb()

    df = df.fillna("")
    df.columns = df.columns.str.strip()

    df["metadata_soup"] = (
        "Title: "          + df["title"].astype(str) +
        " | Genre: "       + df["genres"].astype(str) +
        " | Director: "    + df["director"].astype(str) +
        " | Cast: "        + df["cast"].astype(str) +
        " | Year: "        + df["year"].astype(str) +
        " | Description: " + df["overview"].astype(str)
    )
    return df


# ──────────────────────────────────────────────────────────────
# 3. Build FAISS index on startup
# ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = load_dataframe()
print(f"  {len(df)} movies loaded.")

print("Loading sentence-transformer model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding movies and building FAISS index...")
movie_embeddings = embed_model.encode(
    df["metadata_soup"].tolist(),
    show_progress_bar=True,
    batch_size=64,
).astype("float32")

dimension = movie_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(movie_embeddings)
print(f"FAISS index ready — {index.ntotal} movies indexed.")


# ──────────────────────────────────────────────────────────────
# 4. Retrieval
# ──────────────────────────────────────────────────────────────
def get_movie_recommendations(query: str, k: int = 5) -> pd.DataFrame:
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = distances[0]
    desired   = ["title", "genres", "director", "rating", "overview", "year", "cast"]
    available = [c for c in desired if c in results.columns]
    return results[available]


# ──────────────────────────────────────────────────────────────
# 5. Response generation with conversation memory
# ──────────────────────────────────────────────────────────────
def generate_response(user_query: str, chat_history: list) -> str:
    retrieved = get_movie_recommendations(user_query, k=3)

    movie_context = ""
    for _, row in retrieved.iterrows():
        genres   = row.get("genres",   "N/A")
        overview = row.get("overview", "")
        year     = str(row.get("year", "")).strip()
        director = str(row.get("director", "")).strip()
        rating   = str(row.get("rating",   "")).strip()
        movie_context += (
            f"- {row['title']}"
            f" ({year + ', ' if year else ''}{genres})"
            f"{' | Dir: ' + director if director else ''}"
            f"{' | Rating: ' + rating if rating else ''}"
            f": {overview}\n"
        )

    history_str = ""
    for turn in (chat_history or [])[-4:]:
        if isinstance(turn, dict):
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        + (f"Conversation so far:\n{history_str}\n" if history_str else "")
        + f"User: {user_query}\n\n"
        f"Retrieved Movies:\n{movie_context}\n\n"
        f"Assistant:"
    )

    for attempt in range(3):
        try:
            response = model_llm.generate_content(full_prompt)
            return response.text
        except google_exceptions.TooManyRequests:
            wait = 30 * (attempt + 1)
            print(f"Quota exceeded — waiting {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
        except Exception as e:
            return f"Error: {e}"

    return "Quota exceeded after 3 retries. Please wait a moment and try again."


# ──────────────────────────────────────────────────────────────
# 6. Gradio UI
# ──────────────────────────────────────────────────────────────
CSS = """
#mm-header    { text-align: center; padding: 1.8rem 0 .6rem; }
#mm-header h1 { font-size: 2.6rem; font-weight: 900; letter-spacing: -1.5px;
                background: linear-gradient(135deg, #6366f1, #a855f7);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#mm-header p  { color: #9ca3af; font-size: .95rem; margin-top: .3rem; }
#mm-footer    { text-align: center; color: #6b7280; font-size: .75rem;
                margin-top: 1rem; padding-bottom: .5rem; }
footer        { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        font=[gr.themes.GoogleFont("Syne"), "ui-sans-serif", "sans-serif"],
    ),
    title="MovieMate",
    css=CSS,
) as demo:

    with gr.Column(elem_id="mm-header"):
        gr.HTML("<h1>🎬 MovieMate</h1>")
        gr.HTML(
            "<p>Conversational AI for intelligent movie search &amp; recommendations"
            " — powered by FAISS + Gemini 1.5 Flash</p>"
        )

    chatbot = gr.Chatbot(
        label="MovieMate",
        type="messages",
        height=500,
        show_label=False,
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/microsoft/319/clapper-board_1f3ac.png",
        ),
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Ask me about movies...  e.g. 'Recommend a thriller like Se7en'",
            show_label=False,
            scale=9,
            container=False,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    gr.Examples(
        label="Try these",
        examples=[
            "Recommend a sci-fi movie like Inception",
            "Best drama movies with high ratings",
            "Who directed Interstellar?",
            "Movies starring Leonardo DiCaprio",
            "Something feel-good for a Friday night",
            "Top-rated action movies from 2023",
        ],
        inputs=txt,
    )

    gr.HTML(
        "<div id='mm-footer'>"
        "SentenceTransformers · FAISS · Gemini 1.5 Flash · Gradio"
        " &nbsp;|&nbsp; 980 movies indexed"
        "</div>"
    )

    def respond(message, history):
        if not message.strip():
            return "", history or []
        history = history or []
        reply   = generate_response(message, history)
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history

    txt.submit(respond, [txt, chatbot], [txt, chatbot])
    send_btn.click(respond, [txt, chatbot], [txt, chatbot])

if __name__ == "__main__":
    demo.launch()
