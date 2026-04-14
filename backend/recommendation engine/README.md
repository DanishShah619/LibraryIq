# LibraIQ ML Service

Semantic book recommendation microservice for LibraIQ.

## What it does

Accepts a free-text query (e.g. "a dark mystery with unexpected twists") plus optional
category and tone filters, and returns a ranked list of book recommendations with
confidence scores.

## Stack

- **FastAPI** — REST API
- **ChromaDB** — vector store for semantic search
- **all-MiniLM-L6-v2** — HuggingFace sentence embedding model
- **Redis** — response caching (shared with the Next.js app)

---

## Quick Start (Docker)

```bash
# From the project root
docker compose up --build ml
```

The service starts on `http://localhost:8000`.

First start builds the ChromaDB index (~60–90s). Subsequent starts load from the
persisted `chroma_data` volume (~1s).

---

## API Reference

All endpoints except `/health` require:
```
Authorization: Bearer <ML_SERVICE_SECRET>
```

### `GET /health`
No auth required. Returns service status.

```json
{ "status": "ok", "books_loaded": 5197, "cache_available": true }
```

### `GET /meta`
Returns available filter options for the UI.

```json
{
  "categories": ["All", "Fiction", "Mystery", ...],
  "tones": ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"],
  "book_count": 5197
}
```

### `POST /predict`
Main recommendation endpoint.

**Request:**
```json
{
  "query": "a story about forgiveness and redemption",
  "category": "Fiction",
  "tone": "Sad",
  "top_k": 16
}
```

**Validation rules:**
- `query`: required, 1–500 characters
- `category`: must be "All" or a valid category from `/meta`
- `tone`: must be one of `All | Happy | Surprising | Angry | Suspenseful | Sad`
- `top_k`: integer 1–50 (clamped automatically)

**Response:**
```json
{
  "recommendations": [
    {
      "isbn13": "9780743273565",
      "title": "The Great Gatsby",
      "authors": "F. Scott Fitzgerald",
      "categories": "Fiction",
      "thumbnail": "http://...",
      "published_year": 1925,
      "average_rating": 3.9,
      "description": "A mysterious millionaire pursues...",
      "confidence": 0.923,
      "emotions": {
        "joy": 0.21,
        "surprise": 0.18,
        "anger": 0.09,
        "fear": 0.12,
        "sadness": 0.84
      }
    }
  ],
  "count": 16,
  "query": "a story about forgiveness and redemption",
  "category": "Fiction",
  "tone": "Sad",
  "cached": false,
  "latency_ms": 312.4
}
```

### `POST /admin/reindex`
Force a full ChromaDB index rebuild. Use after adding new books to the dataset CSV.
Automatically clears the Redis recommendation cache.

```json
{ "status": "ok", "books_indexed": 5197 }
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ML_SERVICE_SECRET` | Yes | — | Shared secret for Authorization header |
| `REDIS_URL` | No | `redis://redis:6379` | Redis connection string |
| `CACHE_TTL_SECONDS` | No | `3600` | Cache TTL in seconds |
| `DATA_PATH` | No | `data/books_with_emotions.csv` | Path to books dataset |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | HuggingFace embedding model name |

---

## Data Preprocessing

Before deploying, run the preprocessing script once to fix emotion scores
(recomputes using MEAN instead of MAX aggregation):

```bash
cd ml
pip install transformers torch pandas numpy tqdm

# Full recompute (requires GPU or takes a long time on CPU)
python scripts/recompute_emotions.py

# Quick cleanup only (null fills, drop unused columns) — no GPU needed
python scripts/recompute_emotions.py --skip-emotion
```

---

## Running Tests

```bash
cd ml
pip install pytest httpx
pytest tests/ -v
```

---

## Integration with Next.js

The Next.js app proxies all ML requests — members never call this service directly (FR-ML-03).

Copy `nextjs_proxy/recommendations_ml_route.ts` to:
```
src/app/api/recommendations/ml/route.ts
```

This proxy:
1. Authenticates the user via NextAuth session
2. Forwards the request to the ML service with the shared secret
3. Resolves `isbn13` values to LibraIQ `cuid` book IDs (Improvement 14)
4. Persists the recommendation to the `MLRecommendation` table
5. Returns the merged response to the client

**Important:** To maximise the number of ML recommendations that resolve to
LibraIQ books, bulk-import `data/books_with_emotions.csv` via:
```
POST /api/books/import
```

Without this, only the 5 seed books will match and most ML recommendations will
be filtered out.

---

## Architecture Notes

```
User browser
    │
    ▼
Next.js (port 3000)
    │  POST /api/recommendations/ml
    │  (auth check + isbn13 → cuid resolution)
    ▼
ML Service (port 8000)
    │  similarity_search_with_score
    ▼
ChromaDB (persisted to chroma_data volume)
    │
    ▼
Redis cache (shared with Next.js)
```
