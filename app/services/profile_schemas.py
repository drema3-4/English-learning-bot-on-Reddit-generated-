from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CEFRLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2", "unknown"]
Confidence = Literal["low", "medium", "high"]
Priority = Literal["low", "medium", "high"]

_ALLOWED_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2", "unknown"}
_ALLOWED_PRIORITIES = {"low", "medium", "high"}
_MAX_LIST_ITEMS = 10
_MAX_LIST_ITEM_CHARS = 160
_MAX_LONG_TEXT_CHARS = 1000


class LearningProfilePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_default=True)

    cefr_level: CEFRLevel = "unknown"
    level_confidence: Confidence = "low"
    goals_summary: str = "No specific goals provided."
    focus_areas: list[str] = Field(default_factory=list)
    domain_interests: list[str] = Field(default_factory=list)
    preferred_item_types: dict[str, Priority] = Field(default_factory=dict)
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    difficulty_policy: str = "Prefer practical items appropriate for the user's level."
    extraction_guidance: str = (
        "Prioritize useful reusable language grounded in the source text."
    )

    @field_validator("cefr_level", mode="before")
    @classmethod
    def _normalize_level(cls, value: object) -> str:
        if not isinstance(value, str):
            return "unknown"
        normalized = " ".join(value.split()).upper()
        if normalized in _ALLOWED_LEVELS:
            return normalized
        return "unknown"

    @field_validator("level_confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> str:
        normalized = _normalize_text(value).lower()
        if normalized in _ALLOWED_PRIORITIES:
            return normalized
        return "low"

    @field_validator("goals_summary", "difficulty_policy", "extraction_guidance", mode="before")
    @classmethod
    def _normalize_long_text(cls, value: object) -> str:
        normalized = _truncate(_normalize_text(value), _MAX_LONG_TEXT_CHARS)
        if normalized:
            return normalized
        return "No specific guidance provided."

    @field_validator("focus_areas", "domain_interests", "include", "exclude", mode="before")
    @classmethod
    def _normalize_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]
        items: list[str] = []
        for item in raw_items:
            normalized = _truncate(_normalize_text(item), _MAX_LIST_ITEM_CHARS)
            if normalized and normalized not in items:
                items.append(normalized)
            if len(items) >= _MAX_LIST_ITEMS:
                break
        return items

    @field_validator("preferred_item_types", mode="before")
    @classmethod
    def _normalize_priorities(cls, value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}

        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _truncate(_normalize_text(raw_key).lower(), 64)
            priority = _normalize_text(raw_value).lower()
            if not key:
                continue
            if priority not in _ALLOWED_PRIORITIES:
                priority = "medium"
            normalized[key] = priority
            if len(normalized) >= _MAX_LIST_ITEMS:
                break
        return normalized


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()
