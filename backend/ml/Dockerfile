FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Removed static build-time training because DB is disconnected during build

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Train model at runtime when DB is accessible, then start server
CMD bash -c "python train.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"
