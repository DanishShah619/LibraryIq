"""
Automated Route A DB Synchronization script.
Dynamically syncs scikit-learn models across active Postgres clusters.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from sqlalchemy import create_engine

def train():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("[train] DATABASE_URL not found in container environment. Aborting DB sync.")
        return

    # SQLAlchemy dialects demand postgresql:// over the legacy prisma shorthand postgres://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    print(f"[train] Initializing SQLAlchemy network bridge to Postgres Engine...")
    try:
        engine = create_engine(db_url)
        query = '''
            SELECT 
                b.isbn, 
                b.title, 
                array_to_string(b.authors, ' | ') as authors,
                b.synopsis as description,
                string_agg(g.name, ' | ') as genres
            FROM "Book" b
            LEFT JOIN "BookGenre" bg ON b.id = bg."bookId"
            LEFT JOIN "Genre" g ON bg."genreId" = g.id
            WHERE b."isDeleted" = false
            GROUP BY b.id
        '''
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"[train] FATAL DB ERROR: Could not connect to internal Postgres network. {e}")
        return

    if len(df) == 0:
        print("[train] WARNING: The database is completely empty. No books to train on.")
        return

    print(f"[train] Postgres payload loaded successfully. Active rows hijacked: {len(df)}.")

    # Flatten empty SQL cells
    df.fillna("", inplace=True)

    # Concatenate the dense keyword payload for mathematics
    df["text"] = df["title"] + " " + df["authors"] + " " + df["genres"] + " " + df["description"]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    similarity = cosine_similarity(tfidf_matrix)

    # Cache payload 
    catalogue = df[["isbn", "title", "authors"]].to_dict("records")

    model = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "similarity": similarity,
        "catalogue": catalogue,
    }

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    joblib.dump(model, "model.joblib")
    print(f"[train] 🔥 TF-IDF Engine Successfully Compiled! Library index size: {len(df)}")

if __name__ == "__main__":
    train()
