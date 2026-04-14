# ── LibraIQ ML Service ────────────────────────────────────────────────────────
#
# Improvement 13: Dockerfile bakes the embedding model in at build time.
# This eliminates cold-start download delays and removes runtime HuggingFace dependency.

FROM python:3.11-slim

WORKDIR /app

# System dependencies required by sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Improvement 13: pre-download embedding model at BUILD TIME
# This bakes the model weights into the image layer so:
#   1. First request is fast — no download on startup
#   2. Service works even if HuggingFace Hub is unreachable at runtime
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading all-MiniLM-L6-v2 model...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model downloaded successfully.')"

# Copy application code
COPY . .

# Create data directory (actual data files come via Docker volume or COPY)
RUN mkdir -p data chroma_db

EXPOSE 8000

# Health check — start_period gives ChromaDB time to load on first start
# Improvement 3: subsequent starts are fast because ChromaDB loads from persisted volume
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
