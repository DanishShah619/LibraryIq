"""
Preprocessing script — run once locally before deploying.

What it does:
  1. Recomputes emotion scores using MEAN across sentences (was MAX — see impl plan)
  2. Drops 'disgust' and 'neutral' columns (unused in tone filtering)
  3. Fills null authors/categories with sensible defaults
  4. Saves the cleaned dataset to data/books_with_emotions.csv

Requirements:
  pip install transformers torch pandas numpy tqdm

Run:
  python scripts/recompute_emotions.py

NOTE: This requires a GPU or will be very slow on CPU (5197 books × ~5 sentences each).
If you don't want to rerun the emotion model, use --skip-emotion to only apply
the cleanup steps (null fills, column drops) without recomputing scores.

Usage:
  python scripts/recompute_emotions.py                # full recompute
  python scripts/recompute_emotions.py --skip-emotion # cleanup only
"""

import argparse
import sys
import numpy as np
import pandas as pd

DATA_PATH = "data/books_with_emotions.csv"
OUTPUT_PATH = "data/books_with_emotions.csv"
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def cleanup_only(books: pd.DataFrame) -> pd.DataFrame:
    """Apply non-emotion fixes: fill nulls, drop unused columns."""
    print("Filling null authors...")
    books["authors"] = books["authors"].fillna("Unknown Author")

    print("Filling null categories...")
    books["categories"] = books["categories"].fillna("Uncategorized")

    print("Filling null thumbnails...")
    books["thumbnail"] = books["thumbnail"].fillna("")

    print("Dropping unused emotion columns (disgust, neutral)...")
    books = books.drop(columns=["disgust", "neutral"], errors="ignore")

    return books


def recompute_emotions(books: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute emotion scores using MEAN across sentences instead of MAX.
    This gives a more representative score for the book's overall emotional tone.
    """
    try:
        from transformers import pipeline
        from tqdm import tqdm
    except ImportError:
        print("ERROR: transformers and tqdm required. Run: pip install transformers tqdm")
        sys.exit(1)

    print("Loading emotion classifier (j-hartmann/emotion-english-distilroberta-base)...")
    try:
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=0,  # use GPU if available; change to -1 for CPU
        )
    except Exception:
        print("GPU not available, falling back to CPU (will be slow)...")
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,
        )

    isbn_list = []
    emotion_scores = {label: [] for label in EMOTION_LABELS}

    def calculate_mean_scores(predictions):
        """Average emotion scores across all sentences — more representative than max."""
        per_emotion = {label: [] for label in EMOTION_LABELS}
        for prediction in predictions:
            sorted_preds = sorted(prediction, key=lambda x: x["label"])
            for idx, label in enumerate(EMOTION_LABELS):
                per_emotion[label].append(sorted_preds[idx]["score"])
        # MEAN instead of MAX — this is the core fix
        return {label: float(np.mean(scores)) for label, scores in per_emotion.items()}

    print(f"Recomputing emotion scores for {len(books)} books using MEAN aggregation...")
    for i in tqdm(range(len(books))):
        row = books.iloc[i]
        isbn_list.append(row["isbn13"])

        description = str(row["description"])
        sentences = [s.strip() for s in description.split(".") if s.strip()]
        if not sentences:
            sentences = [description]

        try:
            predictions = classifier(sentences)
            scores = calculate_mean_scores(predictions)
        except Exception as e:
            print(f"\nWarning: failed on isbn {row['isbn13']}: {e}")
            # fallback: zero scores
            scores = {label: 0.0 for label in EMOTION_LABELS}

        for label in EMOTION_LABELS:
            emotion_scores[label].append(scores[label])

    emotions_df = pd.DataFrame(emotion_scores)
    emotions_df["isbn13"] = isbn_list

    # Drop old emotion columns and merge new ones
    books = books.drop(columns=EMOTION_LABELS, errors="ignore")
    books = pd.merge(books, emotions_df, on="isbn13", how="left")

    return books


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-emotion",
        action="store_true",
        help="Skip emotion recomputation, only apply cleanup fixes",
    )
    args = parser.parse_args()

    print(f"Loading {DATA_PATH}...")
    books = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(books)} books with columns: {books.columns.tolist()}")

    if args.skip_emotion:
        print("\nSkipping emotion recomputation (--skip-emotion flag set)")
        print("Applying cleanup fixes only...")
        books = cleanup_only(books)
    else:
        print("\nRecomputing emotion scores with MEAN aggregation...")
        books = recompute_emotions(books)
        books = cleanup_only(books)

    print(f"\nSaving to {OUTPUT_PATH}...")
    books.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Final shape: {books.shape}")
    print(f"Columns: {books.columns.tolist()}")

    # Sanity check
    print("\nEmotion score ranges after processing:")
    for col in ["anger", "fear", "joy", "sadness", "surprise"]:
        if col in books.columns:
            print(f"  {col}: mean={books[col].mean():.3f}, std={books[col].std():.3f}")


if __name__ == "__main__":
    main()
