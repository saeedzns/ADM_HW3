from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Book Search Engine Demo", page_icon="📚", layout="wide")

DATA_PATH = Path("articles.tsv")


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    # fallback partial match
    for col in df.columns:
        c = col.lower()
        if any(candidate.lower() in c for candidate in candidates):
            return col
    return None


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, sep="\t", dtype=str, keep_default_na=False)
    else:
        # Small fallback so the app still runs before the large TSV is uploaded.
        df = pd.DataFrame(
            [
                {
                    "bookTitle": "The Hobbit",
                    "bookAuthors": "J.R.R. Tolkien",
                    "Plot": "A hobbit joins dwarves on an adventure involving a dragon, treasure, courage and friendship.",
                    "ratingValue": "4.3",
                    "Url": "https://www.goodreads.com/",
                },
                {
                    "bookTitle": "Dune",
                    "bookAuthors": "Frank Herbert",
                    "Plot": "Politics, ecology, religion and power collide on a desert planet with valuable spice.",
                    "ratingValue": "4.2",
                    "Url": "https://www.goodreads.com/",
                },
                {
                    "bookTitle": "Pride and Prejudice",
                    "bookAuthors": "Jane Austen",
                    "Plot": "A sharp social novel about manners, marriage, class and misunderstanding.",
                    "ratingValue": "4.3",
                    "Url": "https://www.goodreads.com/",
                },
            ]
        )
    return df.fillna("")


@st.cache_resource(show_spinner=False)
def build_search_index(df: pd.DataFrame, text_col: str):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=1,
        max_features=60000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(df[text_col].astype(str))
    return vectorizer, matrix


def render_result(row: pd.Series, score: float, columns: dict[str, Optional[str]]):
    title = row.get(columns["title"], "Untitled") if columns["title"] else "Untitled"
    authors = row.get(columns["authors"], "Unknown author") if columns["authors"] else "Unknown author"
    plot = row.get(columns["plot"], "") if columns["plot"] else ""
    rating = row.get(columns["rating"], "") if columns["rating"] else ""
    url = row.get(columns["url"], "") if columns["url"] else ""

    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(f"Author(s): {authors} · TF-IDF similarity: {score:.3f}" + (f" · Rating: {rating}" if rating else ""))
        if plot:
            st.write(plot[:900] + ("..." if len(plot) > 900 else ""))
        if isinstance(url, str) and url.startswith("http"):
            st.link_button("Open source page", url)


st.title("📚 Book Search Engine Demo")
st.caption("Interactive demo for Saeed Zohoorian's ADM_HW3 search-engine project")

st.markdown(
    """
This app turns the original notebook/search-engine homework into a usable demo.
It loads the scraped books dataset, builds a TF-IDF index, and ranks books by query similarity.
"""
)

df = load_data()

columns = {
    "title": find_column(df, ["bookTitle", "title", "book title"]),
    "authors": find_column(df, ["bookAuthors", "author", "authors"]),
    "plot": find_column(df, ["Plot", "description", "summary"]),
    "rating": find_column(df, ["ratingValue", "rating", "average stars"]),
    "reviews": find_column(df, ["reviewCount", "reviews"]),
    "url": find_column(df, ["Url", "URL", "link"]),
}

text_parts = []
for key in ["title", "authors", "plot"]:
    col = columns[key]
    if col:
        text_parts.append(df[col].astype(str))

if not text_parts:
    st.error("Could not find usable text columns in articles.tsv.")
    st.stop()

df = df.copy()
df["_search_text"] = text_parts[0]
for part in text_parts[1:]:
    df["_search_text"] += " " + part

with st.sidebar:
    st.header("Search settings")
    query = st.text_input("Search query", value="fantasy adventure dragon")
    top_k = st.slider("Number of results", 3, 25, 10)
    st.divider()
    st.metric("Books loaded", f"{len(df):,}")
    st.write("Detected columns:")
    st.json(columns)

vectorizer, matrix = build_search_index(df, "_search_text")

if query.strip():
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).ravel()
    ranked_idx = scores.argsort()[::-1][:top_k]

    st.subheader(f"Top results for: `{query}`")
    for idx in ranked_idx:
        render_result(df.iloc[idx], float(scores[idx]), columns)
else:
    st.info("Type a search query to rank books.")

st.divider()
st.markdown(
    """
### What this project demonstrates

- Web scraping and dataset construction.
- Text preprocessing and search ranking.
- TF-IDF vectorization and cosine similarity.
- Turning notebooks into a usable web app.
"""
)
