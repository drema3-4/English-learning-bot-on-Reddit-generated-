from __future__ import annotations

from app.bot.messages import format_sources_status
from app.config import Settings
from app.services.sources import (
    SourceAvailabilityService,
    SourceDetectionStatus,
    SourceDetector,
    SourceType,
)


def test_source_detector_detects_manual_text() -> None:
    result = SourceDetector().detect("  This is a plain English paragraph.  ")

    assert result.status == SourceDetectionStatus.DETECTED
    assert result.source_type == SourceType.MANUAL_TEXT
    assert result.source_ref is None
    assert result.input_text == "This is a plain English paragraph."


def test_source_detector_detects_reddit_post_url() -> None:
    result = SourceDetector().detect("reddit.com/r/test/comments/abc123/title/")

    assert result.status == SourceDetectionStatus.DETECTED
    assert result.source_type == SourceType.REDDIT_POST
    assert result.source_ref == "https://www.reddit.com/r/test/comments/abc123/title/"


def test_source_detector_rejects_unknown_url() -> None:
    result = SourceDetector().detect("https://example.com/article")

    assert result.status == SourceDetectionStatus.UNKNOWN_SOURCE
    assert result.source_type is None


def test_source_detector_detects_empty_input() -> None:
    result = SourceDetector().detect("   ")

    assert result.status == SourceDetectionStatus.EMPTY_INPUT
    assert result.source_type is None


def test_source_detector_rejects_mixed_link_and_text() -> None:
    result = SourceDetector().detect(
        "Please process https://www.reddit.com/r/test/comments/abc123/title/"
    )

    assert result.status == SourceDetectionStatus.UNKNOWN_SOURCE
    assert result.source_type is None


def test_source_availability_manual_text_is_always_available() -> None:
    service = SourceAvailabilityService(
        Settings(reddit_client_id="", reddit_client_secret="")
    )

    assert service.is_available(SourceType.MANUAL_TEXT)


def test_source_availability_reddit_requires_both_credentials() -> None:
    configured = SourceAvailabilityService(
        Settings(reddit_client_id="client", reddit_client_secret="secret")
    )
    missing_secret = SourceAvailabilityService(
        Settings(reddit_client_id="client", reddit_client_secret="")
    )

    assert configured.is_available(SourceType.REDDIT_POST)
    assert not missing_secret.is_available(SourceType.REDDIT_POST)


def test_sources_status_does_not_show_credentials() -> None:
    service = SourceAvailabilityService(
        Settings(reddit_client_id="client-value", reddit_client_secret="secret-value")
    )

    text = format_sources_status(service.list_sources())

    assert "Текст вручную: настроено" in text
    assert "Reddit: настроено" in text
    assert "client-value" not in text
    assert "secret-value" not in text
