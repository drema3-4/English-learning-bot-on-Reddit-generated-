from app.utils.reddit_url import RedditUrlError, extract_reddit_post_ref, is_reddit_url


def test_extract_reddit_post_ref_from_full_url() -> None:
    ref = extract_reddit_post_ref(
        "https://www.reddit.com/r/EnglishLearning/comments/abc123/example_title/"
    )

    assert ref.post_id == "abc123"
    assert ref.normalized_url == "https://www.reddit.com/r/EnglishLearning/comments/abc123/example_title/"


def test_extract_reddit_post_ref_from_url_without_scheme() -> None:
    ref = extract_reddit_post_ref("reddit.com/r/test/comments/xyz987/title/")

    assert ref.post_id == "xyz987"


def test_extract_reddit_post_ref_rejects_non_reddit_url() -> None:
    try:
        extract_reddit_post_ref("https://example.com/r/test/comments/abc/title/")
    except RedditUrlError:
        return

    raise AssertionError("Expected RedditUrlError")


def test_is_reddit_url_accepts_supported_reddit_hosts() -> None:
    assert is_reddit_url("https://www.reddit.com/r/MachineLearning/comments/abc/test/")
    assert is_reddit_url("https://reddit.com/r/LocalLLaMA/comments/abc/test/")
    assert is_reddit_url("https://old.reddit.com/r/NLP/comments/abc/test/")


def test_is_reddit_url_rejects_other_text() -> None:
    assert not is_reddit_url("https://example.com/r/test/comments/abc123/title/")
    assert not is_reddit_url("https://news.ycombinator.com/item?id=123")
    assert not is_reddit_url("just a message")
