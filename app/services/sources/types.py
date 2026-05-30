from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceType(StrEnum):
    MANUAL_TEXT = "manual_text"
    REDDIT_POST = "reddit_post"


class SourceDetectionStatus(StrEnum):
    DETECTED = "detected"
    UNKNOWN_SOURCE = "unknown_source"
    EMPTY_INPUT = "empty_input"


@dataclass(frozen=True)
class DetectedSource:
    status: SourceDetectionStatus
    source_type: SourceType | None
    source_ref: str | None
    input_text: str
