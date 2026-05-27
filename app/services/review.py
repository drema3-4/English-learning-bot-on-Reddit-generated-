from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeAlias, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Phrase,
    PhraseExample,
    PhraseFunction,
    ReviewSession,
    Rule,
    RuleExample,
    User,
    WordLemma,
    WordSurfaceForm,
    WordUsageNote,
)


SessionType = Literal["words", "phrases", "rules"]
ReviewStatus = Literal["card", "empty", "completed", "not_found", "timeout", "invalid_score"]

SESSION_TYPES = {"words", "phrases", "rules"}
DEFAULT_REVIEW_SESSION_TIMEOUT_SECONDS = 120
REVIEW_SESSION_SIZE = 20
LOW_SCORE_ITEMS_LIMIT = 15


@dataclass(frozen=True)
class WordCard:
    lemma: str
    surface_form: str
    meaning_en: str
    meaning_ru: str
    usage_note: str
    usage_note_translation: str


@dataclass(frozen=True)
class PhraseCard:
    phrase: str
    function: str
    meaning_en: str
    meaning_ru: str
    example: str
    example_translation: str


@dataclass(frozen=True)
class RuleCard:
    rule_en: str
    rule_ru: str
    example: str
    example_translation: str


ReviewCardDTO: TypeAlias = WordCard | PhraseCard | RuleCard


@dataclass(frozen=True)
class ReviewResult:
    status: ReviewStatus
    card: ReviewCardDTO | None = None
    session_type: SessionType | None = None
    review_session_id: int | None = None


@dataclass(frozen=True)
class _ReviewCandidate:
    item: dict[str, int]
    key: tuple[int, ...]
    score_sum: int
    last_repetition: datetime | None
    created_at: datetime | None


class NoReviewItemsError(RuntimeError):
    pass


class ReviewItemNotFoundError(RuntimeError):
    pass


class ReviewService:
    def __init__(
        self,
        session: AsyncSession,
        timeout_seconds: int = DEFAULT_REVIEW_SESSION_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._timeout_seconds = timeout_seconds

    async def create_or_continue_session(
        self,
        user_id: int,
        session_type: str,
    ) -> ReviewSession:
        typed_session_type = _typed_session_type(session_type)
        return await self._create_or_continue_session(
            user_id,
            typed_session_type,
            self._timeout_seconds,
        )

    async def get_current_card(self, session_id: int) -> ReviewCardDTO | None:
        review_session = await self._session.get(ReviewSession, session_id)
        if review_session is None or review_session.status != "active":
            return None

        items = _decode_items(review_session.items)
        if review_session.current_index >= len(items):
            await self._finish_session(review_session, "finished")
            return None

        return await self._get_card_for_item(
            review_session.user_id,
            _typed_session_type(review_session.session_type),
            items[review_session.current_index],
        )

    async def rate_current_card(self, session_id: int, score: int) -> bool:
        if score < 1 or score > 5:
            return False

        review_session = await self._session.get(ReviewSession, session_id)
        if review_session is None or review_session.status != "active":
            return False

        if _is_timed_out(review_session, self._timeout_seconds):
            await self._finish_session(review_session, "timeout")
            return False

        try:
            return await self._rate_session(review_session, score)
        except ReviewItemNotFoundError:
            return False

    async def start_or_continue(
        self,
        user_id: int,
        session_type: str,
        timeout_seconds: int,
    ) -> ReviewResult:
        if session_type not in SESSION_TYPES:
            return ReviewResult(status="not_found")

        try:
            review_session = await self._create_or_continue_session(
                user_id,
                _typed_session_type(session_type),
                timeout_seconds,
            )
        except NoReviewItemsError:
            return ReviewResult(status="empty")

        return await self._result_from_session(review_session)

    async def record_rating(
        self,
        user_id: int,
        session_type: str,
        review_session_id: int,
        score: int,
        timeout_seconds: int | None = None,
    ) -> ReviewResult:
        if session_type not in SESSION_TYPES:
            return ReviewResult(status="not_found")
        if score < 1 or score > 5:
            return ReviewResult(status="invalid_score")

        review_session = await self._session.get(ReviewSession, review_session_id)
        if (
            review_session is None
            or review_session.user_id != user_id
            or review_session.session_type != session_type
            or review_session.status != "active"
        ):
            return ReviewResult(status="not_found")

        if timeout_seconds is not None and _is_timed_out(review_session, timeout_seconds):
            await self._finish_session(review_session, "timeout")
            return ReviewResult(status="timeout")

        try:
            has_next = await self._rate_session(review_session, score)
        except ReviewItemNotFoundError:
            return ReviewResult(status="not_found")

        typed_session_type = _typed_session_type(review_session.session_type)
        if not has_next:
            return ReviewResult(
                status="completed",
                session_type=typed_session_type,
                review_session_id=review_session.review_session_id,
            )

        return await self._result_from_session(review_session)

    async def _create_or_continue_session(
        self,
        user_id: int,
        session_type: SessionType,
        timeout_seconds: int,
    ) -> ReviewSession:
        active_session = await self._get_active_session(user_id, session_type)
        if active_session is not None:
            if _is_timed_out(active_session, timeout_seconds):
                await self._finish_session(active_session, "timeout")
            else:
                active_session.updated_at = datetime.now(UTC)
                await self._session.commit()
                return active_session

        items = await self._build_session_items(user_id, session_type)
        if not items:
            raise NoReviewItemsError

        now = datetime.now(UTC)
        review_session = ReviewSession(
            user_id=user_id,
            session_type=session_type,
            items=json.dumps(items, ensure_ascii=False),
            current_index=0,
            status="active",
            updated_at=now,
        )
        self._session.add(review_session)
        await self._session.commit()
        await self._session.refresh(review_session)
        return review_session

    async def _result_from_session(self, review_session: ReviewSession) -> ReviewResult:
        typed_session_type = _typed_session_type(review_session.session_type)
        items = _decode_items(review_session.items)
        if review_session.current_index >= len(items):
            await self._finish_session(review_session, "finished")
            return ReviewResult(
                status="completed",
                session_type=typed_session_type,
                review_session_id=review_session.review_session_id,
            )

        card = await self._get_card_for_item(
            review_session.user_id,
            typed_session_type,
            items[review_session.current_index],
        )
        if card is None:
            return ReviewResult(
                status="not_found",
                session_type=typed_session_type,
                review_session_id=review_session.review_session_id,
            )
        return ReviewResult(
            status="card",
            card=card,
            session_type=typed_session_type,
            review_session_id=review_session.review_session_id,
        )

    async def _get_active_session(
        self,
        user_id: int,
        session_type: SessionType,
    ) -> ReviewSession | None:
        return await self._session.scalar(
            select(ReviewSession)
            .where(
                ReviewSession.user_id == user_id,
                ReviewSession.session_type == session_type,
                ReviewSession.status == "active",
            )
            .order_by(ReviewSession.updated_at.desc(), ReviewSession.review_session_id.desc())
            .limit(1)
        )

    async def _finish_session(self, review_session: ReviewSession, status: str) -> None:
        now = datetime.now(UTC)
        review_session.status = status
        review_session.finished_at = now
        review_session.updated_at = now
        await self._session.commit()

    async def _build_session_items(
        self,
        user_id: int,
        session_type: SessionType,
    ) -> list[dict[str, int]]:
        if session_type == "words":
            return await self._build_word_items(user_id)
        if session_type == "phrases":
            return await self._build_phrase_items(user_id)
        return await self._build_rule_items(user_id)

    async def _build_word_items(self, user_id: int) -> list[dict[str, int]]:
        result = await self._session.execute(
            select(WordUsageNote, WordSurfaceForm, WordLemma)
            .join(
                WordSurfaceForm,
                WordUsageNote.surface_form_id == WordSurfaceForm.surface_form_id,
            )
            .join(WordLemma, WordSurfaceForm.lemma_id == WordLemma.lemma_id)
            .where(WordLemma.user_id == user_id)
        )
        candidates = [
            _ReviewCandidate(
                item={
                    "lemma_id": lemma.lemma_id,
                    "surface_form_id": surface_form.surface_form_id,
                    "usage_note_id": note.usage_note_id,
                },
                key=(lemma.lemma_id, surface_form.surface_form_id, note.usage_note_id),
                score_sum=(
                    lemma.current_score
                    + surface_form.current_score
                    + note.current_score
                ),
                last_repetition=_oldest_component_repetition(lemma, surface_form, note),
                created_at=_oldest_created_at(lemma, surface_form, note),
            )
            for note, surface_form, lemma in result.all()
        ]
        return _select_review_items(candidates)

    async def _build_phrase_items(self, user_id: int) -> list[dict[str, int]]:
        result = await self._session.execute(
            select(PhraseExample, PhraseFunction, Phrase)
            .join(PhraseFunction, PhraseExample.function_id == PhraseFunction.function_id)
            .join(Phrase, PhraseFunction.phrase_id == Phrase.phrase_id)
            .where(Phrase.user_id == user_id)
        )
        candidates = [
            _ReviewCandidate(
                item={
                    "phrase_id": phrase.phrase_id,
                    "function_id": function.function_id,
                    "example_id": example.example_id,
                },
                key=(phrase.phrase_id, function.function_id, example.example_id),
                score_sum=(
                    phrase.current_score
                    + function.current_score
                    + example.current_score
                ),
                last_repetition=_oldest_component_repetition(phrase, function, example),
                created_at=_oldest_created_at(phrase, function, example),
            )
            for example, function, phrase in result.all()
        ]
        return _select_review_items(candidates)

    async def _build_rule_items(self, user_id: int) -> list[dict[str, int]]:
        result = await self._session.execute(
            select(RuleExample, Rule)
            .join(Rule, RuleExample.rule_id == Rule.rule_id)
            .where(Rule.user_id == user_id)
        )
        candidates = [
            _ReviewCandidate(
                item={
                    "rule_id": rule.rule_id,
                    "example_id": example.example_id,
                },
                key=(rule.rule_id, example.example_id),
                score_sum=rule.current_score + example.current_score,
                last_repetition=_oldest_component_repetition(rule, example),
                created_at=_oldest_created_at(rule, example),
            )
            for example, rule in result.all()
        ]
        return _select_review_items(candidates)

    async def _get_card_for_item(
        self,
        user_id: int,
        session_type: SessionType,
        item: dict[str, int],
    ) -> ReviewCardDTO | None:
        if session_type == "words":
            return await self._get_word_card(user_id, item)
        if session_type == "phrases":
            return await self._get_phrase_card(user_id, item)
        return await self._get_rule_card(user_id, item)

    async def _get_word_card(
        self,
        user_id: int,
        item: dict[str, int],
    ) -> WordCard | None:
        result = await self._session.execute(
            select(WordUsageNote, WordSurfaceForm, WordLemma)
            .join(
                WordSurfaceForm,
                WordUsageNote.surface_form_id == WordSurfaceForm.surface_form_id,
            )
            .join(WordLemma, WordSurfaceForm.lemma_id == WordLemma.lemma_id)
            .where(
                WordUsageNote.usage_note_id == item.get("usage_note_id"),
                WordSurfaceForm.surface_form_id == item.get("surface_form_id"),
                WordLemma.lemma_id == item.get("lemma_id"),
                WordLemma.user_id == user_id,
            )
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None

        note, surface_form, lemma = row
        return WordCard(
            lemma=lemma.lemma,
            surface_form=surface_form.surface_form,
            meaning_en=surface_form.meaning_en,
            meaning_ru=surface_form.meaning_ru,
            usage_note=note.usage_note,
            usage_note_translation=note.usage_note_translation,
        )

    async def _get_phrase_card(
        self,
        user_id: int,
        item: dict[str, int],
    ) -> PhraseCard | None:
        result = await self._session.execute(
            select(PhraseExample, PhraseFunction, Phrase)
            .join(PhraseFunction, PhraseExample.function_id == PhraseFunction.function_id)
            .join(Phrase, PhraseFunction.phrase_id == Phrase.phrase_id)
            .where(
                PhraseExample.example_id == item.get("example_id"),
                PhraseFunction.function_id == item.get("function_id"),
                Phrase.phrase_id == item.get("phrase_id"),
                Phrase.user_id == user_id,
            )
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None

        example, function, phrase = row
        return PhraseCard(
            phrase=phrase.phrase,
            function=function.function,
            meaning_en=function.meaning_en,
            meaning_ru=function.meaning_ru,
            example=example.example,
            example_translation=example.example_translation,
        )

    async def _get_rule_card(
        self,
        user_id: int,
        item: dict[str, int],
    ) -> RuleCard | None:
        result = await self._session.execute(
            select(RuleExample, Rule)
            .join(Rule, RuleExample.rule_id == Rule.rule_id)
            .where(
                RuleExample.example_id == item.get("example_id"),
                Rule.rule_id == item.get("rule_id"),
                Rule.user_id == user_id,
            )
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None

        example, rule = row
        return RuleCard(
            rule_en=rule.rule_en,
            rule_ru=rule.rule_ru,
            example=example.example,
            example_translation=example.example_translation,
        )

    async def _rate_session(self, review_session: ReviewSession, score: int) -> bool:
        items = _decode_items(review_session.items)
        if review_session.current_index >= len(items):
            await self._finish_session(review_session, "finished")
            return False

        item = items[review_session.current_index]
        entities = await self._get_entities_for_item(
            review_session.user_id,
            _typed_session_type(review_session.session_type),
            item,
        )
        if entities is None:
            raise ReviewItemNotFoundError

        now = datetime.now(UTC)
        for entity in entities:
            entity.current_score = score
            entity.last_repetition = now

        review_session.current_index += 1
        review_session.updated_at = now
        if review_session.current_index >= len(items):
            review_session.status = "finished"
            review_session.finished_at = now
            await self._session.commit()
            return False

        await self._session.commit()
        return True

    async def _get_entities_for_item(
        self,
        user_id: int,
        session_type: SessionType,
        item: dict[str, int],
    ) -> tuple[Any, ...] | None:
        if session_type == "words":
            return await self._get_word_entities(user_id, item)
        if session_type == "phrases":
            return await self._get_phrase_entities(user_id, item)
        return await self._get_rule_entities(user_id, item)

    async def _get_word_entities(
        self,
        user_id: int,
        item: dict[str, int],
    ) -> tuple[WordLemma, WordSurfaceForm, WordUsageNote] | None:
        result = await self._session.execute(
            select(WordLemma, WordSurfaceForm, WordUsageNote)
            .join(WordSurfaceForm, WordSurfaceForm.lemma_id == WordLemma.lemma_id)
            .join(WordUsageNote, WordUsageNote.surface_form_id == WordSurfaceForm.surface_form_id)
            .where(
                WordLemma.lemma_id == item.get("lemma_id"),
                WordSurfaceForm.surface_form_id == item.get("surface_form_id"),
                WordUsageNote.usage_note_id == item.get("usage_note_id"),
                WordLemma.user_id == user_id,
            )
            .limit(1)
        )
        return result.one_or_none()

    async def _get_phrase_entities(
        self,
        user_id: int,
        item: dict[str, int],
    ) -> tuple[Phrase, PhraseFunction, PhraseExample] | None:
        result = await self._session.execute(
            select(Phrase, PhraseFunction, PhraseExample)
            .join(PhraseFunction, PhraseFunction.phrase_id == Phrase.phrase_id)
            .join(PhraseExample, PhraseExample.function_id == PhraseFunction.function_id)
            .where(
                Phrase.phrase_id == item.get("phrase_id"),
                PhraseFunction.function_id == item.get("function_id"),
                PhraseExample.example_id == item.get("example_id"),
                Phrase.user_id == user_id,
            )
            .limit(1)
        )
        return result.one_or_none()

    async def _get_rule_entities(
        self,
        user_id: int,
        item: dict[str, int],
    ) -> tuple[Rule, RuleExample] | None:
        result = await self._session.execute(
            select(Rule, RuleExample)
            .join(RuleExample, RuleExample.rule_id == Rule.rule_id)
            .where(
                Rule.rule_id == item.get("rule_id"),
                RuleExample.example_id == item.get("example_id"),
                Rule.user_id == user_id,
            )
            .limit(1)
        )
        return result.one_or_none()


async def get_next_item(
    session: AsyncSession,
    telegram_id: int,
    item_type: str,
) -> ReviewCardDTO | None:
    user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
    session_type = _legacy_item_type_to_session_type(item_type)
    if user is None or session_type is None:
        return None

    result = await ReviewService(session, timeout_seconds=0).start_or_continue(
        user.user_id,
        session_type,
        timeout_seconds=0,
    )
    return result.card


async def record_rating(
    session: AsyncSession,
    telegram_id: int,
    target: str,
    item_id: int,
    rating: int,
) -> bool:
    user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
    if user is None:
        return False

    result = await ReviewService(session).record_rating(user.user_id, target, item_id, rating)
    return result.status in {"card", "completed"}


def format_item(item: ReviewCardDTO) -> str:
    from app.bot.messages import format_review_card

    return format_review_card(item)


def _select_review_items(candidates: list[_ReviewCandidate]) -> list[dict[str, int]]:
    low_score_candidates = sorted(candidates, key=_low_score_sort_key)[:LOW_SCORE_ITEMS_LIMIT]

    selected: list[_ReviewCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in low_score_candidates:
        if candidate.key not in seen:
            selected.append(candidate)
            seen.add(candidate.key)

    for candidate in sorted(candidates, key=_old_repetition_sort_key):
        if len(selected) >= REVIEW_SESSION_SIZE:
            break
        if candidate.key in seen:
            continue
        selected.append(candidate)
        seen.add(candidate.key)

    return [candidate.item for candidate in selected]


def _low_score_sort_key(candidate: _ReviewCandidate) -> tuple[Any, ...]:
    return (
        candidate.score_sum,
        *_old_repetition_sort_key(candidate),
    )


def _old_repetition_sort_key(candidate: _ReviewCandidate) -> tuple[Any, ...]:
    last_repetition = _normalize_datetime(candidate.last_repetition)
    return (
        last_repetition is not None,
        last_repetition or _datetime_floor(),
        _normalize_datetime(candidate.created_at) or _datetime_floor(),
        candidate.score_sum,
        candidate.key,
    )


def _oldest_component_repetition(*entities: Any) -> datetime | None:
    repetitions = [getattr(entity, "last_repetition", None) for entity in entities]
    if not repetitions or any(repetition is None for repetition in repetitions):
        return None
    return min(cast(datetime, _normalize_datetime(repetition)) for repetition in repetitions)


def _oldest_created_at(*entities: Any) -> datetime | None:
    created_at_values = [
        _normalize_datetime(getattr(entity, "created_at", None))
        for entity in entities
        if getattr(entity, "created_at", None) is not None
    ]
    if not created_at_values:
        return None
    return min(cast(datetime, created_at) for created_at in created_at_values)


def _decode_items(raw_items: str) -> list[dict[str, int]]:
    try:
        value = json.loads(raw_items)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []

    items: list[dict[str, int]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, int] = {}
        for key, raw_value in raw_item.items():
            if not isinstance(key, str):
                continue
            try:
                item[key] = int(raw_value)
            except (TypeError, ValueError):
                continue
        if item:
            items.append(item)
    return items


def _is_timed_out(review_session: ReviewSession, timeout_seconds: int) -> bool:
    if timeout_seconds <= 0:
        return False
    updated_at = _normalize_datetime(review_session.updated_at)
    if updated_at is None:
        return False
    return datetime.now(UTC) - updated_at > timedelta(seconds=timeout_seconds)


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _datetime_floor() -> datetime:
    return datetime.min.replace(tzinfo=UTC)


def _typed_session_type(session_type: str) -> SessionType:
    if session_type == "words":
        return "words"
    if session_type == "phrases":
        return "phrases"
    if session_type == "rules":
        return "rules"
    raise ValueError(f"Unsupported review session type: {session_type}")


def _legacy_item_type_to_session_type(item_type: str) -> SessionType | None:
    if item_type == "word":
        return "words"
    if item_type == "phrase":
        return "phrases"
    if item_type == "rule":
        return "rules"
    if item_type in SESSION_TYPES:
        return _typed_session_type(item_type)
    return None
