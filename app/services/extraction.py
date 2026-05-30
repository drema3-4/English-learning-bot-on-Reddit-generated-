from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Callable, Iterable, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.db.models import LLMExtractionJob
from app.db.session import create_session_factory
from app.services.llm import LLMClient
from app.services.profile_prompts import render_profile_for_prompt
from app.services.profile_schemas import LearningProfilePayload
from app.services.reddit import RedditPost, format_reddit_text
from app.utils.json_parse import JSONParseError, parse_json_array, parse_json_object


MAX_WORDS = 60
MAX_PHRASES = 40
MAX_RULES = 25
EXTRACTION_CHUNK_MAX_CHARS = 10_000
EXTRACTION_CHUNK_OVERLAP_CHARS = 700


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
PromptBuilder = Callable[[str, str, int | None, int | None], str]


WORDS_PROMPT = """Extract useful English vocabulary from the source text.
Return only a valid JSON array. Do not add markdown. Do not add explanations outside JSON.

Rules:
- do not extract articles;
- do not extract pronouns;
- do not extract very simple function words;
- choose important terms and useful study vocabulary;
- account for ML/DL/NLP context when it appears;
- prefer words whose meaning depends on the context.
- Use the user learning profile to decide what is worth extracting.
- Extract all useful words from this source text that match the profile, up to the configured limit.
- Prefer items that are useful for the user's level and goals.
- Do not extract items that the profile explicitly excludes unless they are central to the source text.
- Do not invent examples that are unrelated to the source.

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
"""


PHRASES_PROMPT = """Extract useful English phrases and phrase constructions from the source text.
Return only a valid JSON array. Do not add markdown. Do not add explanations outside JSON.

Look for:
- stable expressions;
- conversational constructions;
- discourse markers;
- typical phrases from discussions;
- useful phrases for argumentation in English.
- Use the user learning profile to prioritize phrases, constructions, discourse markers,
  argumentation patterns, collocations and reusable chunks.

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
"""


RULES_PROMPT = """Extract useful English grammar and usage rules from the source text.
Return only a valid JSON array. Do not add markdown. Do not add explanations outside JSON.

Look for:
- characteristic rules;
- grammatical constructions;
- ways to build phrases;
- useful usage patterns.
- Use the user learning profile to choose grammar and usage rules that are practical
  for this user's level and goals.

Each object must have exactly these keys:
[
  {
    "rule_en": "rule in English",
    "rule_ru": "rule in Russian",
    "example": "example sentence",
    "example_translation": "Russian translation of the example"
  }
]
"""


def build_words_prompt(
    source_text: str,
    profile_prompt: str,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> str:
    return _build_prompt(WORDS_PROMPT, source_text, profile_prompt, chunk_index, chunk_count)


def build_phrases_prompt(
    source_text: str,
    profile_prompt: str,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> str:
    return _build_prompt(PHRASES_PROMPT, source_text, profile_prompt, chunk_index, chunk_count)


def build_rules_prompt(
    source_text: str,
    profile_prompt: str,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> str:
    return _build_prompt(RULES_PROMPT, source_text, profile_prompt, chunk_index, chunk_count)


def split_text_for_extraction(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    source_text = (text or "").strip()
    if not source_text:
        return []
    if max_chars <= 0 or len(source_text) <= max_chars:
        return [source_text]

    pieces = [
        part.strip()
        for part in re.split(r"\n\s*\n|\n", source_text)
        if part.strip()
    ]
    atomic_pieces: list[str] = []
    for piece in pieces:
        atomic_pieces.extend(_split_oversized_piece(piece, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in atomic_pieces:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    return _add_chunk_overlap(chunks, overlap_chars)


def _build_prompt(
    base_prompt: str,
    source_text: str,
    profile_prompt: str,
    chunk_index: int | None,
    chunk_count: int | None,
) -> str:
    parts = [base_prompt.strip(), profile_prompt.strip()]
    if chunk_index is not None and chunk_count is not None and chunk_count > 1:
        parts.append(f"Chunk: {chunk_index} of {chunk_count}.")
    parts.extend(["Source text:", source_text.strip()])
    return "\n\n".join(part for part in parts if part)


def _split_oversized_piece(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", piece)
        if sentence.strip()
    ]
    if len(sentences) > 1:
        return _pack_or_split_units(sentences, max_chars)
    return _split_by_words(piece, max_chars)


def _pack_or_split_units(units: list[str], max_chars: int) -> list[str]:
    result: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_by_words(unit, max_chars))
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            result.append(current)
            current = unit
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def _split_by_words(text: str, max_chars: int) -> list[str]:
    result: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip() if current else word
        if current and len(candidate) > max_chars:
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def _add_chunk_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    with_overlap = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        overlap = _tail_for_overlap(previous, overlap_chars)
        with_overlap.append(f"{overlap}\n\n{current}".strip() if overlap else current)
    return with_overlap


def _tail_for_overlap(text: str, overlap_chars: int) -> str:
    if len(text) <= overlap_chars:
        return text.strip()
    tail = text[-overlap_chars:].strip()
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :].strip()
    return tail


def _profile_prompt_from_snapshot(profile_snapshot: str | None) -> str:
    if not profile_snapshot:
        return "User learning profile:\n- English level: unknown"
    try:
        payload = json.loads(profile_snapshot)
    except json.JSONDecodeError as exc:
        raise ValueError("Learning profile snapshot is not valid JSON") from exc
    profile = LearningProfilePayload.model_validate(payload)
    return render_profile_for_prompt(profile)


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
        max_words: int = MAX_WORDS,
        max_phrases: int = MAX_PHRASES,
        max_rules: int = MAX_RULES,
        chunk_max_chars: int = EXTRACTION_CHUNK_MAX_CHARS,
        chunk_overlap_chars: int = EXTRACTION_CHUNK_OVERLAP_CHARS,
    ) -> None:
        self._session_factory = session_factory
        self._llm_client = llm_client
        self._max_words = max_words
        self._max_phrases = max_phrases
        self._max_rules = max_rules
        self._chunk_max_chars = chunk_max_chars
        self._chunk_overlap_chars = chunk_overlap_chars

    async def extract_words(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        profile_id: int | None = None,
        profile_snapshot: str | None = None,
    ) -> list[WordExtract]:
        return await self._extract(
            user_id=user_id,
            processing_job_id=processing_job_id,
            text=text,
            job_type="words",
            prompt_builder=build_words_prompt,
            model_type=WordExtract,
            limit=self._max_words,
            profile_id=profile_id,
            profile_snapshot=profile_snapshot,
        )

    async def extract_phrases(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        profile_id: int | None = None,
        profile_snapshot: str | None = None,
    ) -> list[PhraseExtract]:
        return await self._extract(
            user_id=user_id,
            processing_job_id=processing_job_id,
            text=text,
            job_type="phrases",
            prompt_builder=build_phrases_prompt,
            model_type=PhraseExtract,
            limit=self._max_phrases,
            profile_id=profile_id,
            profile_snapshot=profile_snapshot,
        )

    async def extract_rules(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        profile_id: int | None = None,
        profile_snapshot: str | None = None,
    ) -> list[RuleExtract]:
        return await self._extract(
            user_id=user_id,
            processing_job_id=processing_job_id,
            text=text,
            job_type="rules",
            prompt_builder=build_rules_prompt,
            model_type=RuleExtract,
            limit=self._max_rules,
            profile_id=profile_id,
            profile_snapshot=profile_snapshot,
        )

    async def _extract(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        job_type: str,
        prompt_builder: PromptBuilder,
        model_type: type[ModelT],
        limit: int,
        profile_id: int | None,
        profile_snapshot: str | None,
    ) -> list[ModelT]:
        profile_prompt = _profile_prompt_from_snapshot(profile_snapshot)
        chunks = split_text_for_extraction(
            text,
            max_chars=self._chunk_max_chars,
            overlap_chars=self._chunk_overlap_chars,
        )
        chunk_count = len(chunks)
        collected: list[ModelT] = []

        for index, chunk in enumerate(chunks, start=1):
            prompt = prompt_builder(chunk, profile_prompt, index, chunk_count)
            llm_job_id = await self._start_job(
                user_id=user_id,
                processing_job_id=processing_job_id,
                profile_id=profile_id,
                profile_snapshot=profile_snapshot,
                job_type=job_type,
                input_text=chunk,
                prompt_text=prompt,
                chunk_index=index,
                chunk_count=chunk_count,
            )
            raw_response: str | None = None
            try:
                raw_response = await self._llm_client.complete_json(prompt)
                parsed = _parse_extraction_items(raw_response, job_type)
                items = [model_type.model_validate(item) for item in parsed]
            except Exception as exc:  # noqa: BLE001
                await self._mark_job_failed(llm_job_id, raw_response, str(exc))
                raise

            await self._mark_job_done(llm_job_id, raw_response, items[:limit])
            collected.extend(items)

        return _dedupe_extracted_items(collected, job_type)[:limit]

    async def _start_job(
        self,
        user_id: int,
        processing_job_id: int,
        profile_id: int | None,
        profile_snapshot: str | None,
        job_type: str,
        input_text: str,
        prompt_text: str,
        chunk_index: int | None,
        chunk_count: int | None,
    ) -> int:
        async with self._session_factory() as session:
            llm_job = LLMExtractionJob(
                user_id=user_id,
                processing_job_id=processing_job_id,
                profile_id=profile_id,
                profile_snapshot=profile_snapshot,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
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


async def extract_words(
    user_id: int,
    processing_job_id: int,
    text: str,
    profile_id: int | None = None,
    profile_snapshot: str | None = None,
) -> list[WordExtract]:
    return await _get_default_service().extract_words(
        user_id,
        processing_job_id,
        text,
        profile_id=profile_id,
        profile_snapshot=profile_snapshot,
    )


async def extract_phrases(
    user_id: int,
    processing_job_id: int,
    text: str,
    profile_id: int | None = None,
    profile_snapshot: str | None = None,
) -> list[PhraseExtract]:
    return await _get_default_service().extract_phrases(
        user_id,
        processing_job_id,
        text,
        profile_id=profile_id,
        profile_snapshot=profile_snapshot,
    )


async def extract_rules(
    user_id: int,
    processing_job_id: int,
    text: str,
    profile_id: int | None = None,
    profile_snapshot: str | None = None,
) -> list[RuleExtract]:
    return await _get_default_service().extract_rules(
        user_id,
        processing_job_id,
        text,
        profile_id=profile_id,
        profile_snapshot=profile_snapshot,
    )


@lru_cache(maxsize=1)
def _get_default_service() -> ExtractionService:
    settings = get_settings()
    return ExtractionService(
        create_session_factory(settings),
        LLMClient(settings),
        max_words=settings.extraction_max_words,
        max_phrases=settings.extraction_max_phrases,
        max_rules=settings.extraction_max_rules,
        chunk_max_chars=settings.extraction_chunk_max_chars,
        chunk_overlap_chars=settings.extraction_chunk_overlap_chars,
    )


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


def _dedupe_extracted_items(items: Iterable[ModelT], job_type: str) -> list[ModelT]:
    deduped: list[ModelT] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        key = _extracted_item_key(item, job_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extracted_item_key(item: BaseModel, job_type: str) -> tuple[str, ...]:
    data = item.model_dump()
    if job_type == "words":
        fields = ("lemma", "meaning_en", "meaning_ru", "usage_note")
    elif job_type == "phrases":
        fields = ("phrase", "function", "meaning_en", "meaning_ru", "example")
    elif job_type == "rules":
        fields = ("rule_en", "rule_ru", "example")
    else:
        fields = tuple(sorted(data))
    return tuple(_normalize_key_part(data.get(field)) for field in fields)


def _normalize_key_part(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).casefold()


def _parse_extraction_items(raw_response: str, job_type: str) -> list[dict[str, Any]]:
    try:
        return parse_json_array(raw_response)
    except JSONParseError as array_error:
        try:
            payload = parse_json_object(raw_response)
        except JSONParseError:
            raise array_error from None

    if _looks_like_extraction_item(payload, job_type):
        return [payload]

    keyed_items = _find_items_by_key(payload, _payload_keys(job_type))
    if keyed_items:
        return keyed_items

    list_values = _collect_json_object_lists(payload)
    if len(list_values) == 1:
        return list_values[0]

    raise JSONParseError(f"Could not find a JSON array for {job_type}")


def _payload_keys(job_type: str) -> set[str]:
    if job_type == "words":
        return _normalize_payload_keys(
            "words",
            "vocabulary",
            "vocab",
            "terms",
            "important_words",
            "useful_words",
            "items",
        )
    if job_type == "phrases":
        return _normalize_payload_keys(
            "phrases",
            "useful_phrases",
            "phrase_constructions",
            "constructions",
            "expressions",
            "stable_expressions",
            "discourse_markers",
            "items",
        )
    if job_type == "rules":
        return _normalize_payload_keys(
            "rules",
            "grammar_rules",
            "usage_rules",
            "grammar",
            "patterns",
            "usage_patterns",
            "items",
        )
    return _normalize_payload_keys("items", job_type)


def _is_json_object_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _find_items_by_key(value: object, keys: set[str]) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            if _normalize_payload_key(raw_key) in keys:
                items = _flatten_json_object_lists(raw_value)
                if items:
                    return items

        for raw_value in value.values():
            items = _find_items_by_key(raw_value, keys)
            if items:
                return items

    if isinstance(value, list):
        for item in value:
            items = _find_items_by_key(item, keys)
            if items:
                return items

    return []


def _flatten_json_object_lists(value: object) -> list[dict[str, Any]]:
    return [
        item
        for item_list in _collect_json_object_lists(value)
        for item in item_list
    ]


def _collect_json_object_lists(value: object) -> list[list[dict[str, Any]]]:
    if _is_json_object_list(value):
        return [value]
    if isinstance(value, dict):
        result: list[list[dict[str, Any]]] = []
        for item in value.values():
            result.extend(_collect_json_object_lists(item))
        return result
    if isinstance(value, list):
        result: list[list[dict[str, Any]]] = []
        for item in value:
            result.extend(_collect_json_object_lists(item))
        return result
    return []


def _looks_like_extraction_item(payload: dict[str, Any], job_type: str) -> bool:
    keys = {_normalize_payload_key(key) for key in payload}
    if job_type == "words":
        return bool({"lemma", "surface_form"} & keys)
    if job_type == "phrases":
        return bool({"phrase", "function", "meaning_en", "meaning_ru"} & keys)
    if job_type == "rules":
        return bool({"rule", "rule_en", "rule_ru"} & keys)
    return False


def _normalize_payload_keys(*keys: str) -> set[str]:
    return {_normalize_payload_key(key) for key in keys}


def _normalize_payload_key(key: object) -> str:
    if not isinstance(key, str):
        return ""
    return "_".join(key.strip().lower().replace("-", "_").split())


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
