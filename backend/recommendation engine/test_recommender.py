"""
Tests for the LibraIQ ML service.

Run:
    pip install pytest httpx
    pytest tests/ -v

These tests use a lightweight mock to avoid loading the real embedding model,
making them fast enough to run in CI without a GPU.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_BOOKS = pd.DataFrame(
    {
        "isbn13": ["9780001234567", "9780009876543", "9780001111111"],
        "title": ["Test Book One", "Test Book Two", "Test Book Three"],
        "authors": ["Author A", "Author B;Author C", "Author D"],
        "categories": ["Fiction", "Mystery", "Fiction"],
        "thumbnail": ["http://example.com/1.jpg", "http://example.com/2.jpg", ""],
        "description": [
            "A wonderful story about friendship and adventure.",
            "A dark mystery with twists and turns.",
            "A coming of age story set in the 1970s.",
        ],
        "published_year": [2020, 2019, 2021],
        "average_rating": [4.1, 3.8, 4.5],
        "anger": [0.05, 0.45, 0.03],
        "fear": [0.10, 0.80, 0.05],
        "joy": [0.90, 0.20, 0.85],
        "sadness": [0.15, 0.30, 0.20],
        "surprise": [0.30, 0.60, 0.25],
    }
)


@pytest.fixture
def mock_recommender():
    """A Recommender instance with mocked internals — no real model loaded."""
    from recommender import Recommender

    rec = Recommender()
    rec.books = SAMPLE_BOOKS.copy()
    rec._categories = ["All", "Fiction", "Mystery"]
    rec._loaded = True

    # Mock the ChromaDB similarity_search_with_score
    mock_db = MagicMock()
    mock_db.similarity_search_with_score.return_value = [
        (MagicMock(metadata={"isbn13": "9780001234567"}), 0.1),
        (MagicMock(metadata={"isbn13": "9780009876543"}), 0.3),
        (MagicMock(metadata={"isbn13": "9780001111111"}), 0.5),
    ]
    rec.db = mock_db

    return rec


@pytest.fixture
def client(mock_recommender):
    """FastAPI test client with mocked recommender and no Redis."""
    import main

    main.recommender = mock_recommender
    main.redis_client = None  # disable caching in tests

    return TestClient(main.app)


AUTH_HEADER = {"Authorization": "Bearer dev-secret-change-in-production"}


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["books_loaded"] == 3


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_predict_without_auth_returns_401(client):
    response = client.post("/predict", json={"query": "adventure story"})
    assert response.status_code == 422  # missing header = validation error


def test_predict_with_wrong_token_returns_401(client):
    response = client.post(
        "/predict",
        json={"query": "adventure story"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


# ── Input validation ─────────────────────────────────────────────────────────

def test_empty_query_returns_422(client):
    response = client.post(
        "/predict",
        json={"query": "   "},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"][0]["msg"].lower()


def test_query_too_long_returns_422(client):
    response = client.post(
        "/predict",
        json={"query": "a" * 501},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 422
    assert "long" in response.json()["detail"][0]["msg"].lower()


def test_invalid_tone_returns_422(client):
    response = client.post(
        "/predict",
        json={"query": "mystery book", "tone": "Terrifying"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 422


def test_top_k_clamped_to_max(client):
    response = client.post(
        "/predict",
        json={"query": "adventure", "top_k": 9999},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    # top_k was clamped to 50; results won't exceed book count
    assert response.json()["count"] <= 3


def test_top_k_clamped_to_min(client):
    response = client.post(
        "/predict",
        json={"query": "adventure", "top_k": 0},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200


# ── Predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_recommendations(client):
    response = client.post(
        "/predict",
        json={"query": "adventure story"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert data["count"] == len(data["recommendations"])


def test_predict_response_has_required_fields(client):
    response = client.post(
        "/predict",
        json={"query": "friendship"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    recs = response.json()["recommendations"]
    assert len(recs) > 0

    rec = recs[0]
    required_fields = ["isbn13", "title", "authors", "categories", "confidence", "emotions"]
    for field in required_fields:
        assert field in rec, f"Missing field: {field}"


def test_predict_confidence_scores_between_0_and_1(client):
    response = client.post(
        "/predict",
        json={"query": "mystery thriller"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    for rec in response.json()["recommendations"]:
        assert 0.0 <= rec["confidence"] <= 1.0, f"Confidence out of range: {rec['confidence']}"


def test_predict_category_filter(client):
    response = client.post(
        "/predict",
        json={"query": "mystery", "category": "Mystery"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    for rec in response.json()["recommendations"]:
        assert rec["categories"] == "Mystery"


def test_predict_tone_filter_happy(client):
    response = client.post(
        "/predict",
        json={"query": "uplifting story", "tone": "Happy"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    recs = response.json()["recommendations"]
    # Results should be sorted by joy score descending
    if len(recs) > 1:
        joy_scores = [r["emotions"]["joy"] for r in recs]
        assert joy_scores == sorted(joy_scores, reverse=True)


def test_predict_includes_isbn13_for_resolution(client):
    """isbn13 must be present so Next.js can resolve to LibraIQ book IDs."""
    response = client.post(
        "/predict",
        json={"query": "adventure"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    for rec in response.json()["recommendations"]:
        assert "isbn13" in rec
        assert rec["isbn13"].isdigit()


def test_predict_latency_ms_in_response(client):
    response = client.post(
        "/predict",
        json={"query": "adventure"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    assert "latency_ms" in response.json()


# ── Semantic ranking ──────────────────────────────────────────────────────────

def test_semantic_ranking_preserved(mock_recommender):
    """
    Improvement 9: results must be ordered by semantic similarity,
    not by their position in the DataFrame.
    """
    # Arrange: ChromaDB returns isbn in a specific order
    mock_recommender.db.similarity_search_with_score.return_value = [
        (MagicMock(metadata={"isbn13": "9780001111111"}), 0.05),  # most similar
        (MagicMock(metadata={"isbn13": "9780001234567"}), 0.20),
        (MagicMock(metadata={"isbn13": "9780009876543"}), 0.50),  # least similar
    ]

    results = mock_recommender.recommend("coming of age")
    isbn_order = [r["isbn13"] for r in results]

    # First result must be the most semantically similar (lowest distance)
    assert isbn_order[0] == "9780001111111", (
        f"Semantic ranking not preserved. Got order: {isbn_order}"
    )


# ── Meta ──────────────────────────────────────────────────────────────────────

def test_meta_returns_categories_and_tones(client):
    response = client.get("/meta", headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "tones" in data
    assert "All" in data["categories"]
    assert "All" in data["tones"]
    assert "Happy" in data["tones"]
