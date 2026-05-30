from __future__ import annotations

from typing import Protocol


class SourceFetcher(Protocol):
    async def fetch_text(self, source_ref: str | None) -> str:
        ...


class RedditPostFetcher:
    def __init__(self, reddit_service: object, comments_limit: int) -> None:
        self._reddit_service = reddit_service
        self._comments_limit = comments_limit

    async def fetch_text(self, source_ref: str | None) -> str:
        if not source_ref:
            raise ValueError("Reddit source_ref is empty")
        return await self._reddit_service.fetch_post_text(
            source_ref,
            comments_limit=self._comments_limit,
        )
