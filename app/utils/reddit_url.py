from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


REDDIT_HOSTS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "new.reddit.com",
}


class RedditUrlError(ValueError):
    pass


@dataclass(frozen=True)
class RedditPostRef:
    post_id: str
    normalized_url: str


def extract_reddit_post_ref(raw_url: str) -> RedditPostRef:
    value = raw_url.strip()
    if not value:
        raise RedditUrlError("Empty Reddit URL")

    if value.startswith(("reddit.com/", "www.reddit.com/", "old.reddit.com/", "new.reddit.com/")):
        value = f"https://{value}"

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise RedditUrlError("URL must use http or https")

    host = parsed.netloc.lower()
    if host not in REDDIT_HOSTS:
        raise RedditUrlError("URL must point to reddit.com")

    parts = [part for part in parsed.path.split("/") if part]
    try:
        comments_index = parts.index("comments")
        post_id = parts[comments_index + 1]
    except (ValueError, IndexError) as exc:
        raise RedditUrlError("URL must point to a Reddit post") from exc

    if not re.fullmatch(r"[A-Za-z0-9]+", post_id):
        raise RedditUrlError("Reddit post id is invalid")

    normalized = urlunparse(("https", "www.reddit.com", parsed.path, "", "", ""))
    return RedditPostRef(post_id=post_id, normalized_url=normalized)


def is_reddit_url(text: str) -> bool:
    try:
        extract_reddit_post_ref(text)
    except RedditUrlError:
        return False
    return True
