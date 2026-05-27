from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import models  # noqa: F401
from app.db.base import Base
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
from app.services.review import PhraseCard, ReviewService, RuleCard, WordCard


@pytest_asyncio.fixture
async def session_factory() -> async_sessionmaker:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


async def test_word_session_uses_low_scores_then_oldest_repetitions(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        session.add(user)
        await session.flush()

        now = datetime.now(UTC)
        usage_note_ids: list[int] = []
        for index in range(22):
            last_repetition = None if index >= 17 else now - timedelta(minutes=index)
            usage_note = _word_item(
                user.user_id,
                index,
                score=index + 1,
                last_repetition=last_repetition,
            )
            session.add(usage_note)
            await session.flush()
            usage_note_ids.append(usage_note.usage_note_id)
        await session.commit()

        review_session = await ReviewService(session).create_or_continue_session(
            user.user_id,
            "words",
        )
        items = json.loads(review_session.items)
        selected_usage_note_ids = [item["usage_note_id"] for item in items]

    assert len(items) == 20
    assert selected_usage_note_ids[:15] == usage_note_ids[:15]
    assert selected_usage_note_ids[15:] == usage_note_ids[17:22]


async def test_word_session_returns_all_items_when_less_than_session_size(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        session.add(user)
        await session.flush()

        for index in range(3):
            session.add(
                _word_item(
                    user.user_id,
                    index,
                    score=1,
                    last_repetition=datetime.now(UTC) - timedelta(days=index),
                )
            )
        await session.commit()

        review_session = await ReviewService(session).create_or_continue_session(
            user.user_id,
            "words",
        )
        items = json.loads(review_session.items)

    assert len(items) == 3


async def test_rate_current_word_card_updates_all_word_scores_and_finishes(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        usage_note = _word_item(user.user_id, 1, score=1, last_repetition=None)
        user.word_lemmas.append(usage_note.surface_form.lemma)
        session.add(user)
        await session.commit()

        service = ReviewService(session)
        review_session = await service.create_or_continue_session(user.user_id, "words")
        card = await service.get_current_card(review_session.review_session_id)
        has_next = await service.rate_current_card(review_session.review_session_id, 5)

        saved_session = await session.get(ReviewSession, review_session.review_session_id)

    assert isinstance(card, WordCard)
    assert card.surface_form == "word-1"
    assert has_next is False
    assert saved_session is not None
    assert saved_session.status == "finished"
    assert usage_note.surface_form.lemma.current_score == 5
    assert usage_note.surface_form.current_score == 5
    assert usage_note.current_score == 5
    assert usage_note.last_repetition is not None


async def test_rate_current_phrase_card_updates_phrase_function_and_example_scores(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        example = _phrase_example(user, index=1)
        session.add(example)
        await session.commit()

        service = ReviewService(session)
        review_session = await service.create_or_continue_session(user.user_id, "phrases")
        has_next = await service.rate_current_card(review_session.review_session_id, 3)

    assert has_next is False
    assert example.function.phrase.current_score == 3
    assert example.function.current_score == 3
    assert example.current_score == 3
    assert example.last_repetition is not None


async def test_rate_current_rule_card_updates_rule_and_example_scores(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        example = _rule_example(user, index=1)
        session.add(example)
        await session.commit()

        service = ReviewService(session)
        review_session = await service.create_or_continue_session(user.user_id, "rules")
        has_next = await service.rate_current_card(review_session.review_session_id, 2)

    assert has_next is False
    assert example.rule.current_score == 2
    assert example.current_score == 2
    assert example.last_repetition is not None


async def test_phrase_and_rule_cards_are_dto_objects(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        phrase_example = PhraseExample(
            example="To be fair, this works.",
            example_translation="Справедливости ради, это работает.",
            function=PhraseFunction(
                function="softens disagreement",
                meaning_en="adds balance",
                meaning_ru="смягчает несогласие",
                phrase=Phrase(phrase="to be fair", user=user),
            ),
        )
        rule_example = RuleExample(
            example="I would say it works.",
            example_translation="Я бы сказал, это работает.",
            rule=Rule(
                rule_en="Use would to soften opinions.",
                rule_ru="Would смягчает мнение.",
                user=user,
            ),
        )
        session.add_all([phrase_example, rule_example])
        await session.commit()

        service = ReviewService(session)
        phrase_session = await service.create_or_continue_session(user.user_id, "phrases")
        rule_session = await service.create_or_continue_session(user.user_id, "rules")
        phrase_card = await service.get_current_card(phrase_session.review_session_id)
        rule_card = await service.get_current_card(rule_session.review_session_id)

    assert phrase_card == PhraseCard(
        phrase="to be fair",
        function="softens disagreement",
        meaning_en="adds balance",
        meaning_ru="смягчает несогласие",
        example="To be fair, this works.",
        example_translation="Справедливости ради, это работает.",
    )
    assert rule_card == RuleCard(
        rule_en="Use would to soften opinions.",
        rule_ru="Would смягчает мнение.",
        example="I would say it works.",
        example_translation="Я бы сказал, это работает.",
    )


def _word_item(
    user_id: int,
    index: int,
    score: int,
    last_repetition: datetime | None,
) -> WordUsageNote:
    return WordUsageNote(
        usage_note=f"Usage note {index}.",
        usage_note_translation=f"Пример {index}.",
        current_score=score,
        last_repetition=last_repetition,
        surface_form=WordSurfaceForm(
            surface_form=f"word-{index}",
            meaning_en=f"meaning en {index}",
            meaning_ru=f"значение {index}",
            current_score=score,
            last_repetition=last_repetition,
            lemma=WordLemma(
                lemma=f"lemma-{index}",
                user_id=user_id,
                current_score=score,
                last_repetition=last_repetition,
            ),
        ),
    )


def _phrase_example(user: User, index: int) -> PhraseExample:
    return PhraseExample(
        example=f"Phrase example {index}.",
        example_translation=f"Phrase example translation {index}.",
        current_score=1,
        function=PhraseFunction(
            function=f"function {index}",
            meaning_en=f"phrase meaning en {index}",
            meaning_ru=f"phrase meaning ru {index}",
            current_score=1,
            phrase=Phrase(
                phrase=f"phrase-{index}",
                current_score=1,
                user=user,
            ),
        ),
    )


def _rule_example(user: User, index: int) -> RuleExample:
    return RuleExample(
        example=f"Rule example {index}.",
        example_translation=f"Rule example translation {index}.",
        current_score=1,
        rule=Rule(
            rule_en=f"Rule en {index}.",
            rule_ru=f"Rule ru {index}.",
            current_score=1,
            user=user,
        ),
    )
