from __future__ import annotations

import re

from app.services.sources.types import DetectedSource, SourceDetectionStatus, SourceType
from app.utils.reddit_url import RedditUrlError, extract_reddit_post_ref


URL_PATTERN = re.compile(
    r"(?i)(?:https?://|www\.)\S+|(?:reddit\.com|old\.reddit\.com|new\.reddit\.com)/\S+"
)


class SourceDetector:
    def detect(self, raw_text: str | None) -> DetectedSource:
        input_text = (raw_text or "").strip()
        if not input_text:
            return DetectedSource(
                status=SourceDetectionStatus.EMPTY_INPUT,
                source_type=None,
                source_ref=None,
                input_text=input_text,
            )

        urls = URL_PATTERN.findall(input_text)
        if not urls:
            return DetectedSource(
                status=SourceDetectionStatus.DETECTED,
                source_type=SourceType.MANUAL_TEXT,
                source_ref=None,
                input_text=input_text,
            )

        if len(urls) != 1 or input_text != urls[0]:
            return DetectedSource(
                status=SourceDetectionStatus.UNKNOWN_SOURCE,
                source_type=None,
                source_ref=None,
                input_text=input_text,
            )

        try:
            reddit_ref = extract_reddit_post_ref(urls[0])
        except RedditUrlError:
            return DetectedSource(
                status=SourceDetectionStatus.UNKNOWN_SOURCE,
                source_type=None,
                source_ref=urls[0],
                input_text=input_text,
            )

        return DetectedSource(
            status=SourceDetectionStatus.DETECTED,
            source_type=SourceType.REDDIT_POST,
            source_ref=reddit_ref.normalized_url,
            input_text=input_text,
        )
