"""
scripts/fetch_dataset.py
────────────────────────
Run this once locally to pre-build movies_updated.csv, then commit it
to your HF Space so the app starts instantly without hitting TMDB.

Usage:
    export TMDB_API_KEY="your-key"
    python scripts/fetch_dataset.py
"""

import os
import time
import requests
import pandas as pd

API_KEY = os.environ.get("TMDB_API_KEY", "")
if not API_KEY:
    raise SystemExit("❌  Set the TMDB_API_KEY environment variable first.")

OUTPUT = "movies_updated.csv"

# ── Step 1: fetch movie list ──────────────────
print("⬇️  Fetching movie list (50 pages)…")
frames = []
for page in range(1, 51):
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/discover/movie",
            params={"api_key": API_KEY, "language": "en", "page": page},
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
        print(f"  Page {page}: {len(results)} movies")
    except Exception as e:
        print(f"  Page {page} error: {e}")
    time.sleep(0.1)

df = pd.concat(frames, ignore_index=True)
print(f"\n✅ {len(df)} movies fetched. Enriching…\n")

# ── Step 2: enrich with genres / cast / director ─
ratings, genres_list, directors, casts, durations, years = [], [], [], [], [], []

for i, movie_id in enumerate(df["id"], 1):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": API_KEY, "append_to_response": "credits"},
            timeout=10,
        )
        data = r.json()
        ratings.append(data.get("vote_average"))
        genres_list.append(", ".join(g["name"] for g in data.get("genres", [])))
        director = next(
            (c["name"] for c in data.get("credits", {}).get("crew", [])
             if c["job"] == "Director"),
            None,
        )
        directors.append(director)
        casts.append(
            ", ".join(c["name"] for c in data.get("credits", {}).get("cast", [])[:5])
        )
        durations.append(data.get("runtime"))
        rd = data.get("release_date", "")
        years.append(rd.split("-")[0] if rd else None)
    except Exception as e:
        print(f"  [{i}] Movie {movie_id} error: {e}")
        ratings.append(None); genres_list.append(None); directors.append(None)
        casts.append(None);   durations.append(None);  years.append(None)

    if i % 100 == 0:
        print(f"  Enriched {i}/{len(df)} movies…")
    time.sleep(0.2)

df["rating"]   = ratings
df["genres"]   = genres_list
df["director"] = directors
df["cast"]     = casts
df["duration"] = durations
df["year"]     = years

df.to_csv(OUTPUT, index=False)
print(f"\n✅ Saved {len(df)} movies to {OUTPUT}")
print("   Now commit this file to your HF Space repo:")
print("   git add movies_updated.csv && git commit -m 'add dataset' && git push")
