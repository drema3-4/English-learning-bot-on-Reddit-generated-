import pytest

import app.services.reddit as reddit_module
from app.config import Settings
from app.services.reddit import (
    RedditPost,
    RedditService,
    collect_visible_comments,
    format_reddit_text,
)


def test_format_reddit_text_uses_guide_layout() -> None:
    text = format_reddit_text(
        RedditPost(
            title="A useful question",
            body="How should I say this?",
            comments=["Say it naturally.", "This also works."],
            permalink="https://www.reddit.com/r/test/comments/abc/title/",
        )
    )

    assert text == (
        "Title:\n"
        "A useful question\n"
        "\n"
        "Post:\n"
        "How should I say this?\n"
        "\n"
        "Comments:\n"
        "1. Say it naturally.\n"
        "2. This also works."
    )


def test_collect_visible_comments_filters_removed_and_empty_comments() -> None:
    comments = collect_visible_comments(
        ["First", "[deleted]", " ", "[removed]", "Second"],
        comments_limit=20,
    )

    assert comments == ["First", "Second"]


def test_format_reddit_text_limits_output_size() -> None:
    text = format_reddit_text(
        RedditPost(
            title="x" * 100,
            body="body",
            comments=[],
            permalink="",
        ),
        max_chars=30,
    )

    assert len(text) <= 30


@pytest.mark.asyncio
async def test_fetch_post_uses_asyncpraw_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_reddit = _FakeReddit()

    def fake_reddit_factory(**kwargs: str) -> _FakeReddit:
        fake_reddit.kwargs = kwargs
        return fake_reddit

    monkeypatch.setattr(reddit_module.asyncpraw, "Reddit", fake_reddit_factory)

    settings = Settings(
        reddit_client_id="client-id",
        reddit_client_secret="client-secret",
        reddit_user_agent="test-agent",
    )
    post = await RedditService(settings).fetch_post(
        "https://www.reddit.com/r/test/comments/abc123/title/",
        comments_limit=1,
    )

    assert fake_reddit.kwargs == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "user_agent": "test-agent",
    }
    assert fake_reddit.submission_url == (
        "https://www.reddit.com/r/test/comments/abc123/title/"
    )
    assert fake_reddit.submission_fetch is False
    assert fake_reddit.submission_obj.comment_sort == "top"
    assert fake_reddit.submission_obj.comment_limit == 1
    assert fake_reddit.submission_obj.loaded is True
    assert fake_reddit.submission_obj.comments.more_removed is True
    assert fake_reddit.closed is True
    assert post == RedditPost(
        title="Async title",
        body="Async body",
        comments=["First visible"],
        permalink="https://www.reddit.com/r/test/comments/abc123/title/",
    )


class _FakeReddit:
    def __init__(self) -> None:
        self.kwargs: dict[str, str] = {}
        self.submission_obj = _FakeSubmission()
        self.submission_url: str | None = None
        self.submission_fetch: bool | None = None
        self.closed = False

    async def submission(
        self,
        *,
        url: str,
        fetch: bool,
    ) -> "_FakeSubmission":
        self.submission_url = url
        self.submission_fetch = fetch
        return self.submission_obj

    async def close(self) -> None:
        self.closed = True


class _FakeSubmission:
    def __init__(self) -> None:
        self.title = "Async title"
        self.selftext = "Async body"
        self.permalink = "/r/test/comments/abc123/title/"
        self.comment_sort: str | None = None
        self.comment_limit: int | None = None
        self.comments = _FakeComments(
            [_FakeComment("First visible"), _FakeComment("Second visible")]
        )
        self.loaded = False

    async def load(self) -> None:
        self.loaded = True


class _FakeComments(list["_FakeComment"]):
    def __init__(self, comments: list["_FakeComment"]) -> None:
        super().__init__(comments)
        self.more_removed = False

    async def replace_more(self, *, limit: int) -> list[object]:
        self.more_removed = limit == 0
        return []


class _FakeComment:
    def __init__(self, body: str) -> None:
        self.body = body
