"""
Application configuration — loaded from environment variables / .env file.
All settings have sensible defaults for local development.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Security — must be set in production
    ml_service_secret: str = "dev-secret-change-in-production"

    # Redis — reuses the same Redis instance as the Next.js app
    redis_url: str = "redis://redis:6379"
    cache_ttl_seconds: int = 3600  # 1 hour

    # Data paths
    data_path: str = "data/books_with_emotions.csv"
    chroma_persist_dir: str = "./chroma_db"

    # Embedding model — all-MiniLM-L6-v2 is fast, small, and accurate enough
    embedding_model: str = "all-MiniLM-L6-v2"

    # Recommendation defaults
    default_initial_top_k: int = 50
    default_final_top_k: int = 16
    oversample_multiplier: int = 4  # used when category/tone filters are active

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
