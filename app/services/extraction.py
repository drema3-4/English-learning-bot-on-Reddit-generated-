from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Iterable, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.db.models import LLMExtractionJob
from app.db.session import create_session_factory
from app.services.llm import LLMClient
from app.services.reddit import RedditPost, format_reddit_text
from app.utils.json_parse import parse_json_array


MAX_WORDS = 30
MAX_PHRASES = 20
MAX_RULES = 15


class WordExtract(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    lemma: str
    surface_form: str
    meaning_en: str
    meaning_ru: str
    usage_note: str
    usage_note_translation: str


class PhraseExtract(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    phrase: str
    function: str
    meaning_en: str
    meaning_ru: str
    example: str
    example_translation: str


class RuleExtract(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    rule_en: str
    rule_ru: str
    example: str
    example_translation: str


class JSONCompleter(Protocol):
    async def complete_json(self, prompt: str) -> str:
        ...


ModelT = TypeVar("ModelT", bound=BaseModel)


WORDS_PROMPT = """Extract useful English vocabulary from the source text.
Return only a valid JSON array. Do not add markdown. Do not add explanations outside JSON.

Rules:
- do not extract articles;
- do not extract pronouns;
- do not extract very simple function words;
- choose important terms and useful study vocabulary;
- account for ML/DL/NLP context when it appears;
- prefer words whose meaning depends on the context.

Each object must have exactly these keys:
[
  {
    "lemma": "initial form of the word",
    "surface_form": "word form from the text",
    "meaning_en": "meaning in English in this context",
    "meaning_ru": "meaning in Russian in this context",
    "usage_note": "example sentence from the text or close to the text",
    "usage_note_translation": "Russian translation of the example"
  }
]

Source text:
"""


PHRASES_PROMPT = """Extract useful English phrases and phrase constructions from the source text.
Return only a valid JSON array. Do not add markdown. Do not add explanations outside JSON.

Look for:
- stable expressions;
- conversational constructions;
- discourse markers;
- typical phrases from discussions;
- useful phrases for argumentation in English.

Each object must have exactly these keys:
[
  {
    "phrase": "general phrase construction",
    "function": "how the phrase is used",
    "meaning_en": "meaning of this function in English",
    "meaning_ru": "meaning of this function in Russian",
    "example": "example sentence",
    "example_translation": "Russian translation of the example"
  }
]

Source text:
"""


RULES_PROMPT = """Extract useful English grammar and usage rules from the source text.
Return only a valid JSON array. Do not add markdown. Do not add explanations outside JSON.

Look for:
- characteristic rules;
- grammatical constructions;
- ways to build phrases;
- useful usage patterns.

Each object must have exactly these keys:
[
  {
    "rule_en": "rule in English",
    "rule_ru": "rule in Russian",
    "example": "example sentence",
    "example_translation": "Russian translation of the example"
  }
]

Source text:
"""


@dataclass(frozen=True)
class LearningItemData:
    item_type: str
    text: str
    translation: str | None = None
    explanation: str | None = None
    example: str | None = None
    source_context: str | None = None
    lemma: str | None = None
    meaning_en: str | None = None
    meaning_ru: str | None = None
    function: str | None = None
    rule_ru: str | None = None
    example_translation: str | None = None


class ExtractionService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        llm_client: JSONCompleter,
    ) -> None:
        self._session_factory = session_factory
        self._llm_client = llm_client

    async def extract_words(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[WordExtract]:
        return await self._extract(
            user_id=user_id,
            processing_job_id=processing_job_id,
            text=text,
            job_type="words",
            prompt=WORDS_PROMPT + text,
            model_type=WordExtract,
            limit=MAX_WORDS,
        )

    async def extract_phrases(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[PhraseExtract]:
        return await self._extract(
            user_id=user_id,
            processing_job_id=processing_job_id,
            text=text,
            job_type="phrases",
            prompt=PHRASES_PROMPT + text,
            model_type=PhraseExtract,
            limit=MAX_PHRASES,
        )

    async def extract_rules(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[RuleExtract]:
        return await self._extract(
            user_id=user_id,
            processing_job_id=processing_job_id,
            text=text,
            job_type="rules",
            prompt=RULES_PROMPT + text,
            model_type=RuleExtract,
            limit=MAX_RULES,
        )

    async def _extract(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        job_type: str,
        prompt: str,
        model_type: type[ModelT],
        limit: int,
    ) -> list[ModelT]:
        llm_job_id = await self._start_job(user_id, processing_job_id, job_type, text, prompt)
        raw_response: str | None = None
        try:
            raw_response = await self._llm_client.complete_json(prompt)
            parsed = parse_json_array(raw_response)[:limit]
            items = [model_type.model_validate(item) for item in parsed]
        except Exception as exc:  # noqa: BLE001
            await self._mark_job_failed(llm_job_id, raw_response, str(exc))
            raise

        await self._mark_job_done(llm_job_id, raw_response, items)
        return items

    async def _start_job(
        self,
        user_id: int,
        processing_job_id: int,
        job_type: str,
        input_text: str,
        prompt_text: str,
    ) -> int:
        async with self._session_factory() as session:
            llm_job = LLMExtractionJob(
                user_id=user_id,
                processing_job_id=processing_job_id,
                job_type=job_type,
                input_text=input_text,
                prompt_text=prompt_text,
                status="processing",
                started_at=datetime.now(UTC),
            )
            session.add(llm_job)
            await session.commit()
            await session.refresh(llm_job)
            return llm_job.llm_job_id

    async def _mark_job_done(
        self,
        llm_job_id: int,
        raw_response: str,
        items: list[BaseModel],
    ) -> None:
        async with self._session_factory() as session:
            llm_job = await session.get(LLMExtractionJob, llm_job_id)
            if llm_job is None:
                return

            llm_job.status = "done"
            llm_job.raw_response = raw_response
            llm_job.parsed_response = json.dumps(
                [item.model_dump() for item in items],
                ensure_ascii=False,
            )
            llm_job.error_message = None
            llm_job.finished_at = datetime.now(UTC)
            await session.commit()

    async def _mark_job_failed(
        self,
        llm_job_id: int,
        raw_response: str | None,
        error_message: str,
    ) -> None:
        async with self._session_factory() as session:
            llm_job = await session.get(LLMExtractionJob, llm_job_id)
            if llm_job is None:
                return

            llm_job.status = "failed"
            llm_job.raw_response = raw_response
            llm_job.error_message = error_message[:4000]
            llm_job.finished_at = datetime.now(UTC)
            await session.commit()


def build_source_text(post: RedditPost) -> str:
    return format_reddit_text(post)


def build_learning_items(
    words: Iterable[WordExtract],
    phrases: Iterable[PhraseExtract],
    rules: Iterable[RuleExtract],
) -> list[LearningItemData]:
    items: list[LearningItemData] = []
    items.extend(
        LearningItemData(
            item_type="word",
            text=word.surface_form,
            lemma=word.lemma,
            meaning_en=word.meaning_en,
            meaning_ru=word.meaning_ru,
            translation=word.meaning_ru,
            explanation=word.meaning_en,
            example=word.usage_note,
            example_translation=word.usage_note_translation,
        )
        for word in words
    )
    items.extend(
        LearningItemData(
            item_type="phrase",
            text=phrase.phrase,
            function=phrase.function,
            meaning_en=phrase.meaning_en,
            meaning_ru=phrase.meaning_ru,
            translation=phrase.meaning_ru,
            explanation=phrase.function,
            example=phrase.example,
            example_translation=phrase.example_translation,
        )
        for phrase in phrases
    )
    items.extend(
        LearningItemData(
            item_type="rule",
            text=rule.rule_en,
            rule_ru=rule.rule_ru,
            translation=rule.rule_ru,
            example=rule.example,
            example_translation=rule.example_translation,
        )
        for rule in rules
    )
    return _dedupe_learning_items(items)


def normalize_llm_payload(payload: dict[str, Any]) -> list[LearningItemData]:
    items: list[LearningItemData] = []
    items.extend(_collect_items(payload.get("words"), "word"))
    items.extend(_collect_items(payload.get("phrases"), "phrase"))
    items.extend(_collect_rules(payload.get("rules")))
    return _dedupe_learning_items(items)


async def extract_words(user_id: int, processing_job_id: int, text: str) -> list[WordExtract]:
    return await _get_default_service().extract_words(user_id, processing_job_id, text)


async def extract_phrases(user_id: int, processing_job_id: int, text: str) -> list[PhraseExtract]:
    return await _get_default_service().extract_phrases(user_id, processing_job_id, text)


async def extract_rules(user_id: int, processing_job_id: int, text: str) -> list[RuleExtract]:
    return await _get_default_service().extract_rules(user_id, processing_job_id, text)


@lru_cache(maxsize=1)
def _get_default_service() -> ExtractionService:
    settings = get_settings()
    return ExtractionService(create_session_factory(settings), LLMClient(settings))


def _collect_items(raw_items: object, item_type: str) -> Iterable[LearningItemData]:
    if not isinstance(raw_items, list):
        return []

    collected: list[LearningItemData] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        text = _clean_first(
            raw.get("text"),
            raw.get(item_type),
            raw.get("surface_form"),
            raw.get("lemma"),
            raw.get("phrase"),
        )
        if not text:
            continue
        meaning_en = _clean_first(raw.get("meaning_en"), raw.get("explanation"), raw.get("function"))
        meaning_ru = _clean_first(raw.get("meaning_ru"), raw.get("translation"))
        collected.append(
            LearningItemData(
                item_type=item_type,
                text=text,
                translation=meaning_ru,
                explanation=meaning_en,
                example=_clean_first(raw.get("example"), raw.get("usage_note")),
                source_context=_clean(raw.get("source_context")),
                lemma=_clean(raw.get("lemma")),
                meaning_en=meaning_en,
                meaning_ru=meaning_ru,
                function=_clean(raw.get("function")),
                example_translation=_clean_first(
                    raw.get("example_translation"),
                    raw.get("usage_note_translation"),
                ),
            )
        )
    return collected


def _collect_rules(raw_items: object) -> Iterable[LearningItemData]:
    if not isinstance(raw_items, list):
        return []

    collected: list[LearningItemData] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        text = _clean_first(raw.get("title"), raw.get("text"), raw.get("rule"), raw.get("rule_en"))
        if not text:
            continue
        rule_ru = _clean_first(raw.get("rule_ru"), raw.get("translation"))
        collected.append(
            LearningItemData(
                item_type="rule",
                text=text,
                translation=rule_ru,
                explanation=_clean(raw.get("explanation")),
                example=_clean_first(raw.get("example"), raw.get("usage_note")),
                source_context=_clean(raw.get("source_context")),
                rule_ru=rule_ru,
                example_translation=_clean_first(
                    raw.get("example_translation"),
                    raw.get("usage_note_translation"),
                ),
            )
        )
    return collected


def _dedupe_learning_items(items: Iterable[LearningItemData]) -> list[LearningItemData]:
    deduped: list[LearningItemData] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.item_type, item.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _clean_first(*values: object) -> str | None:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return None
