
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)

# Valid filter values — used by API validation
VALID_TONES = {"All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"}

# Tone → emotion column mapping
TONE_TO_EMOTION = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


class Recommender:
    """
    Encapsulates the full recommendation pipeline.
    Call load() once at startup; then call recommend() per request.
    """

    def __init__(self):
        self.books: Optional[pd.DataFrame] = None
        self.db: Optional[Chroma] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._categories: Optional[list] = None
        self._loaded = False

    # ── Startup ──────────────────────────────────────────────────────────────

    def load(self):
        """
        Load all resources once at application startup.
        Subsequent requests use the already-warm model and index.
        """
        logger.info("Loading books dataset from %s ...", settings.data_path)
        self.books = pd.read_csv(settings.data_path)

        # Improvement 11: drop unused emotion columns
        self.books = self.books.drop(columns=["disgust", "neutral"], errors="ignore")

        # Fill nulls so filters and responses don't break
        self.books["authors"] = self.books["authors"].fillna("Unknown Author")
        self.books["categories"] = self.books["categories"].fillna("Uncategorized")
        self.books["thumbnail"] = self.books["thumbnail"].fillna("")

        # Normalise isbn13 to string for consistent lookup
        self.books["isbn13"] = self.books["isbn13"].astype(str)

        # Cache sorted category list for the /categories endpoint
        self._categories = ["All"] + sorted(
            self.books["categories"].astype(str).unique().tolist()
        )

        logger.info("Loaded %d books.", len(self.books))

        # Improvement 2: load embedding model once at startup
        logger.info("Loading embedding model: %s ...", settings.embedding_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        logger.info("Embedding model loaded.")

        # Improvement 3: load or build ChromaDB index
        self._load_or_build_index()

        self._loaded = True
        logger.info("Recommender ready.")

    def _load_or_build_index(self):
        """Load the persisted ChromaDB index or build it if it doesn't exist."""
        persist_dir = settings.chroma_persist_dir

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            logger.info("Loading existing ChromaDB index from %s ...", persist_dir)
            self.db = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
            logger.info(
                "ChromaDB index loaded (%d documents).",
                self.db._collection.count(),
            )
        else:
            logger.info(
                "No existing index found. Building ChromaDB index for %d books...",
                len(self.books),
            )
            documents = self._build_documents()
            self.db = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=persist_dir,
            )
            logger.info(
                "ChromaDB index built and persisted to %s (%d documents).",
                persist_dir,
                len(documents),
            )

    def _build_documents(self) -> list[Document]:
        """
        Build LangChain Document objects from the books DataFrame.

        Improvement 7: ISBN stored as metadata, NOT embedded in document text.
        This eliminates the fragile 'split on first token' extraction pattern
        and makes the content pure description text for better semantic search.

        Improvement 6: No chunking — each book is one document.
        chunk_size was 250 before which split descriptions across multiple docs,
        causing duplicates and inflated retrieval scores.
        """
        documents = []
        for _, row in self.books.iterrows():
            doc = Document(
                page_content=str(row["description"]),
                metadata={"isbn13": str(row["isbn13"])},
            )
            documents.append(doc)
        logger.info("Built %d documents.", len(documents))
        return documents

    def rebuild_index(self):
        """
        Force a full index rebuild. Called by POST /admin/reindex.
        Use when new books are added to the dataset.
        """
        import shutil

        persist_dir = settings.chroma_persist_dir
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            logger.info("Cleared existing ChromaDB index at %s.", persist_dir)

        logger.info("Rebuilding ChromaDB index...")
        documents = self._build_documents()
        self.db = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=persist_dir,
        )
        logger.info("Index rebuilt with %d documents.", len(documents))

    # ── Recommendation ───────────────────────────────────────────────────────

    def recommend(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        final_top_k: int = 16,
    ) -> list[dict]:
        """
        Return a ranked list of book recommendations with confidence scores.

        Args:
            query:       Free-text description or genre keywords.
            category:    Category filter. "All" disables filtering.
            tone:        Emotional tone filter. "All" disables filtering.
            final_top_k: Number of results to return (1–50).

        Returns:
            List of dicts, each representing a book with a confidence score.
        """
        if not self._loaded:
            raise RuntimeError("Recommender not loaded. Call load() first.")

        initial_top_k = settings.default_initial_top_k

        # Improvement 10: oversample when filters are active
        # Category + tone filtering reduces the result pool significantly.
        # Fetching 4× more candidates ensures we still have enough after filtering.
        filters_active = (category and category != "All") or (tone and tone != "All")
        fetch_k = initial_top_k * settings.oversample_multiplier if filters_active else initial_top_k

        # Semantic search with scores
        # Improvement 5: use similarity_search_with_score to get distances
        raw_results = self.db.similarity_search_with_score(query, k=fetch_k)

        if not raw_results:
            return []

        # Improvement 5: normalise L2 distances → confidence scores (0–1, higher = better)
        distances = [score for _, score in raw_results]
        max_dist = max(distances) if max(distances) > 0 else 1.0
        confidences = [round(1.0 - (d / max_dist), 4) for d in distances]

        # Improvement 7: extract ISBN from metadata (not text parsing)
        isbn_data = [
            (doc.metadata["isbn13"], conf)
            for (doc, _), conf in zip(raw_results, confidences)
        ]

        # Improvement 9: build rank and confidence maps before DataFrame lookup
        # .isin() does NOT preserve order — we re-apply ranking after the lookup
        isbn_rank = {isbn: i for i, (isbn, _) in enumerate(isbn_data)}
        isbn_conf = {isbn: conf for isbn, conf in isbn_data}

        # Filter DataFrame to matching books
        book_recs = self.books[self.books["isbn13"].isin(isbn_rank.keys())].copy()

        # Improvement 9: restore semantic ranking
        book_recs["_rank"] = book_recs["isbn13"].map(isbn_rank)
        book_recs["confidence"] = book_recs["isbn13"].map(isbn_conf)
        book_recs = book_recs.sort_values("_rank").drop(columns=["_rank"])

        # Apply category filter
        if category and category != "All":
            book_recs = book_recs[book_recs["categories"] == category]

        # Apply tone filter
        # Improvement 8: emotion scores were recomputed with MEAN (see preprocessing script)
        if tone and tone in TONE_TO_EMOTION:
            emotion_col = TONE_TO_EMOTION[tone]
            if emotion_col in book_recs.columns:
                book_recs = book_recs.sort_values(by=emotion_col, ascending=False)

        # Take top final_top_k
        book_recs = book_recs.head(final_top_k)

        return self._format_results(book_recs)

    def _format_results(self, book_recs: pd.DataFrame) -> list[dict]:
        """Serialise DataFrame rows to response dicts."""
        results = []
        for _, row in book_recs.iterrows():
            # Format authors string
            authors_raw = str(row.get("authors", "Unknown Author"))
            authors_split = [a.strip() for a in authors_raw.split(";") if a.strip()]
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors_raw

            # Truncate description for the response payload
            description = str(row.get("description", ""))
            truncated_description = (
                " ".join(description.split()[:50]) + "..."
                if len(description.split()) > 50
                else description
            )

            results.append(
                {
                    "isbn13": str(row["isbn13"]),
                    "title": str(row.get("title", "")),
                    "authors": authors_str,
                    "categories": str(row.get("categories", "Uncategorized")),
                    "thumbnail": str(row.get("thumbnail", "")),
                    "published_year": (
                        int(row["published_year"])
                        if pd.notna(row.get("published_year"))
                        else None
                    ),
                    "average_rating": (
                        float(row["average_rating"])
                        if pd.notna(row.get("average_rating"))
                        else None
                    ),
                    "description": truncated_description,
                    "confidence": float(row.get("confidence", 0.0)),
                    # Emotion scores — useful for the Next.js app to display tone badges
                    "emotions": {
                        "joy": round(float(row.get("joy", 0.0)), 4),
                        "surprise": round(float(row.get("surprise", 0.0)), 4),
                        "anger": round(float(row.get("anger", 0.0)), 4),
                        "fear": round(float(row.get("fear", 0.0)), 4),
                        "sadness": round(float(row.get("sadness", 0.0)), 4),
                    },
                }
            )

        return results

    # ── Metadata helpers ─────────────────────────────────────────────────────

    @property
    def categories(self) -> list[str]:
        """Sorted list of available categories, including 'All'."""
        return self._categories or []

    @property
    def tones(self) -> list[str]:
        """Available tone filter values."""
        return ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    @property
    def book_count(self) -> int:
        return len(self.books) if self.books is not None else 0
