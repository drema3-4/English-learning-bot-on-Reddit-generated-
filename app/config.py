from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    telegram_bot_token: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "english-source-learning-bot/0.1"
    database_url: str = "sqlite+aiosqlite:///./data/bot.db"
    max_users: int = 5
    reddit_comments_limit: int = 20
    processing_job_timeout_seconds: int = 120
    review_session_timeout_seconds: int = 120
    profile_required: bool = True
    profile_generation_max_input_chars: int = 3000
    extraction_max_words: int = 60
    extraction_max_phrases: int = 40
    extraction_max_rules: int = 25
    extraction_chunk_max_chars: int = 10_000
    extraction_chunk_overlap_chars: int = 700

    @property
    def has_reddit_credentials(self) -> bool:
        return bool(self.reddit_client_id.strip() and self.reddit_client_secret.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
