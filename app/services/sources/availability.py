from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.sources.types import SourceType


@dataclass(frozen=True)
class SourceAvailability:
    source_type: SourceType
    is_configured: bool
    display_name: str
    unavailable_message: str


class SourceAvailabilityService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_available(self, source_type: SourceType) -> bool:
        return self.get_source(source_type).is_configured

    def get_source(self, source_type: SourceType) -> SourceAvailability:
        for source in self.list_sources():
            if source.source_type == source_type:
                return source
        raise ValueError(f"Unsupported source type: {source_type}")

    def list_sources(self) -> list[SourceAvailability]:
        return [
            SourceAvailability(
                source_type=SourceType.MANUAL_TEXT,
                is_configured=True,
                display_name="Текст вручную",
                unavailable_message="",
            ),
            SourceAvailability(
                source_type=SourceType.REDDIT_POST,
                is_configured=self._settings.has_reddit_credentials,
                display_name="Reddit",
                unavailable_message="Reddit API сейчас не настроен. Пришли текст поста вручную.",
            ),
        ]
