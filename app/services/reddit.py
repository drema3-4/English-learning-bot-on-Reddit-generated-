from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import asyncpraw
import asyncprawcore
from asyncpraw.exceptions import AsyncPRAWException

from app.config import Settings
from app.utils.reddit_url import extract_reddit_post_ref


DEFAULT_COMMENTS_LIMIT = 20
MAX_REDDIT_TEXT_CHARS = 25_000
REMOVED_COMMENT_BODIES = {"[deleted]", "[removed]"}


class RedditFetchError(RuntimeError):
    pass


class RedditCommentLike(Protocol):
    body: str


@dataclass(frozen=True)
class RedditPost:
    title: str
    body: str
    comments: list[str]
    permalink: str


class RedditService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch_post_text(
        self,
        url: str,
        comments_limit: int = DEFAULT_COMMENTS_LIMIT,
    ) -> str:
        post = await self.fetch_post(url, comments_limit=comments_limit)
        return format_reddit_text(post)

    async def fetch_post(
        self,
        reddit_url: str,
        comments_limit: int | None = None,
    ) -> RedditPost:
        limit = self._resolve_comments_limit(comments_limit)
        if not self._settings.has_reddit_credentials:
            raise RedditFetchError("Reddit credentials are missing")

        ref = extract_reddit_post_ref(reddit_url)
        reddit = asyncpraw.Reddit(
            client_id=self._settings.reddit_client_id,
            client_secret=self._settings.reddit_client_secret,
            user_agent=self._settings.reddit_user_agent,
        )

        try:
            submission = await reddit.submission(url=ref.normalized_url, fetch=False)
            submission.comment_sort = "top"
            submission.comment_limit = limit
            await submission.load()

            await submission.comments.replace_more(limit=0)
            comments = collect_visible_comments(submission.comments, limit)
            permalink = _clean_text(getattr(submission, "permalink", ""))
            return RedditPost(
                title=_clean_text(submission.title),
                body=_clean_text(submission.selftext),
                comments=comments,
                permalink=f"https://www.reddit.com{permalink}" if permalink else "",
            )
        except asyncprawcore.AsyncPrawcoreException as exc:
            raise RedditFetchError(f"Reddit API error: {exc}") from exc
        except AsyncPRAWException as exc:
            raise RedditFetchError(f"Reddit client error: {exc}") from exc
        finally:
            await reddit.close()

    def _resolve_comments_limit(self, comments_limit: int | None) -> int:
        raw_limit = (
            self._settings.reddit_comments_limit
            if comments_limit is None
            else comments_limit
        )
        return max(0, raw_limit)


class RedditClient(RedditService):
    pass


def format_reddit_text(
    post: RedditPost,
    max_chars: int = MAX_REDDIT_TEXT_CHARS,
) -> str:
    lines = [
        "Title:",
        post.title.strip(),
        "",
        "Post:",
        post.body.strip(),
        "",
        "Comments:",
    ]
    lines.extend(
        f"{index}. {comment.strip()}"
        for index, comment in enumerate(post.comments, start=1)
        if comment.strip()
    )
    return _limit_text("\n".join(lines), max_chars)


def collect_visible_comments(
    comments: Iterable[RedditCommentLike | str],
    comments_limit: int,
) -> list[str]:
    visible: list[str] = []
    for comment in comments:
        if len(visible) >= comments_limit:
            break
        body = _comment_body(comment)
        if body is not None:
            visible.append(body)
    return visible


def _comment_body(comment: RedditCommentLike | str) -> str | None:
    if isinstance(comment, str):
        body = comment
    else:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            return None

    cleaned = _clean_text(body)
    if not cleaned or cleaned.casefold() in REMOVED_COMMENT_BODIES:
        return None
    return cleaned


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _limit_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()
