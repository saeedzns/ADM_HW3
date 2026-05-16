# ADM_HW3 Streamlit book-search demo

This is the most immediately deployable demo because your repo already contains `articles.tsv`.

## How to use

Copy these files to the root of your `ADM_HW3` repo:

```text
streamlit_app.py
requirements.txt
```

Then deploy on Streamlit Community Cloud with:

```text
Main file path: streamlit_app.py
```

## What it does

- Loads `articles.tsv`.
- Detects title, author, plot, rating, review, and URL columns where possible.
- Builds a TF-IDF matrix over book title + author + plot text.
- Lets the user search for books interactively.
- Returns ranked results with scores and metadata.

## Why this is good for your portfolio

This turns an old university homework into a real data product. That matters more than leaving it as notebooks only.
