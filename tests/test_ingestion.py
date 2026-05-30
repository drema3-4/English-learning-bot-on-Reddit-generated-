from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import (
    Phrase,
    PhraseExample,
    PhraseFunction,
    ProcessingJob,
    Rule,
    RuleExample,
    User,
    WordLemma,
    WordSurfaceForm,
    WordUsageNote,
)
from app.services.extraction import PhraseExtract, RuleExtract, WordExtract
from app.services.ingestion import IngestionService
from app.services.profile_schemas import LearningProfilePayload
from app.services.profiles import ProfileService
from app.services.sources.types import SourceType


class FakeRedditService:
    def __init__(self, text: str = "Reddit source text") -> None:
        self.text = text
        self.calls: list[tuple[str, int]] = []

    async def fetch_post_text(self, url: str, comments_limit: int = 20) -> str:
        self.calls.append((url, comments_limit))
        return self.text


class FailingRedditService:
    async def fetch_post_text(self, url: str, comments_limit: int = 20) -> str:
        raise AssertionError("Reddit API must not be called for manual text jobs")


class FakeExtractionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, str, int | None, str | None]] = []

    async def extract_words(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        profile_id: int | None,
        profile_snapshot: str | None,
    ) -> list[WordExtract]:
        self.calls.append(("words", user_id, processing_job_id, text, profile_id, profile_snapshot))
        return [
            WordExtract(
                lemma="Notice",
                surface_form="noticed",
                meaning_en="became aware of",
                meaning_ru="noticed in Russian",
                usage_note="I noticed the pattern.",
                usage_note_translation="I noticed the pattern in Russian.",
            )
        ]

    async def extract_phrases(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        profile_id: int | None,
        profile_snapshot: str | None,
    ) -> list[PhraseExtract]:
        self.calls.append(("phrases", user_id, processing_job_id, text, profile_id, profile_snapshot))
        return [
            PhraseExtract(
                phrase="to be fair",
                function="softens disagreement",
                meaning_en="adds balance before an argument",
                meaning_ru="to be fair in Russian",
                example="To be fair, the model is small.",
                example_translation="To be fair example in Russian.",
            )
        ]

    async def extract_rules(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
        profile_id: int | None,
        profile_snapshot: str | None,
    ) -> list[RuleExtract]:
        self.calls.append(("rules", user_id, processing_job_id, text, profile_id, profile_snapshot))
        return [
            RuleExtract(
                rule_en="Use would to soften opinions.",
                rule_ru="Would softens opinions in Russian.",
                example="I would say this is overfitting.",
                example_translation="Would example in Russian.",
            )
        ]


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


async def test_process_job_saves_raw_text_and_adds_extracted_items(
    session_factory: async_sessionmaker,
) -> None:
    user_id, job_id = await _create_processing_job(session_factory)
    reddit_service = FakeRedditService("Title and comments")
    extraction_service = FakeExtractionService()
    service = IngestionService(
        session_factory,
        reddit_service,
        extraction_service,
        comments_limit=20,
    )

    await service.process_job(job_id)

    async with session_factory() as session:
        job = await session.get(ProcessingJob, job_id)
        word_lemmas = (await session.scalars(select(WordLemma))).all()
        word_surface_forms = (await session.scalars(select(WordSurfaceForm))).all()
        word_usage_notes = (await session.scalars(select(WordUsageNote))).all()
        phrases = (await session.scalars(select(Phrase))).all()
        phrase_functions = (await session.scalars(select(PhraseFunction))).all()
        phrase_examples = (await session.scalars(select(PhraseExample))).all()
        rules = (await session.scalars(select(Rule))).all()
        rule_examples = (await session.scalars(select(RuleExample))).all()

    assert reddit_service.calls == [
        ("https://www.reddit.com/r/test/comments/abc123/title/", 20)
    ]
    assert job is not None
    assert job.raw_text == "Title and comments"
    assert job.profile_id is not None
    assert job.profile_snapshot is not None
    assert all(call[4] == job.profile_id for call in extraction_service.calls)
    assert all(call[5] == job.profile_snapshot for call in extraction_service.calls)
    assert [lemma.lemma for lemma in word_lemmas] == ["notice"]
    assert len(word_surface_forms) == 1
    assert len(word_usage_notes) == 1
    assert len(phrases) == 1
    assert len(phrase_functions) == 1
    assert len(phrase_examples) == 1
    assert len(rules) == 1
    assert len(rule_examples) == 1
    assert user_id == word_lemmas[0].user_id


async def test_process_manual_text_job_uses_saved_raw_text(
    session_factory: async_sessionmaker,
) -> None:
    _, job_id = await _create_manual_processing_job(session_factory)
    extraction_service = FakeExtractionService()
    service = IngestionService(
        session_factory,
        FailingRedditService(),
        extraction_service,
        comments_limit=20,
    )

    await service.process_job(job_id)

    async with session_factory() as session:
        job = await session.get(ProcessingJob, job_id)

    assert job is not None
    assert job.source_type == SourceType.MANUAL_TEXT
    assert job.source_ref is None
    assert job.raw_text == "Manual English post text"
    assert [call[0] for call in extraction_service.calls] == ["words", "phrases", "rules"]
    assert {call[3] for call in extraction_service.calls} == {"Manual English post text"}
    assert all(call[4] == job.profile_id for call in extraction_service.calls)
    assert all(call[5] == job.profile_snapshot for call in extraction_service.calls)


async def test_process_job_without_profile_snapshot_fails(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        job = ProcessingJob(
            user=user,
            source_type=SourceType.MANUAL_TEXT.value,
            source_ref=None,
            raw_text="Manual English post text",
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.processing_job_id

    service = IngestionService(
        session_factory,
        FailingRedditService(),
        FakeExtractionService(),
        comments_limit=20,
    )

    try:
        await service.process_job(job_id)
    except ValueError as exc:
        assert str(exc) == "Processing job has no learning profile snapshot"
    else:
        raise AssertionError("Expected profile snapshot validation error")


async def test_add_words_reuses_existing_lemma_surface_form_and_usage_note(
    session_factory: async_sessionmaker,
) -> None:
    user_id, _ = await _create_processing_job(session_factory)
    service = IngestionService(
        session_factory,
        FakeRedditService(),
        FakeExtractionService(),
    )
    word = WordExtract(
        lemma="Notice",
        surface_form="noticed",
        meaning_en="became aware of",
        meaning_ru="noticed in Russian",
        usage_note="I noticed the pattern.",
        usage_note_translation="I noticed the pattern in Russian.",
    )

    await service.add_words(user_id, [word, word])

    async with session_factory() as session:
        assert len((await session.scalars(select(WordLemma))).all()) == 1
        assert len((await session.scalars(select(WordSurfaceForm))).all()) == 1
        assert len((await session.scalars(select(WordUsageNote))).all()) == 1

    second_note = word.model_copy(update={"usage_note": "She noticed another clue."})
    await service.add_words(user_id, [second_note])

    async with session_factory() as session:
        usage_notes = (await session.scalars(select(WordUsageNote))).all()

    assert sorted(note.usage_note for note in usage_notes) == [
        "I noticed the pattern.",
        "She noticed another clue.",
    ]


async def _create_processing_job(
    session_factory: async_sessionmaker,
) -> tuple[int, int]:
    async with session_factory() as session:
        user = User(telegram_id=100)
        session.add(user)
        await session.flush()
        profile = await _create_profile(session, user.user_id)
        job = ProcessingJob(
            user=user,
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/abc123/title/",
            profile_id=profile.profile_id,
            profile_snapshot=profile.profile_json,
        )
        session.add(job)
        await session.commit()
        await session.refresh(user)
        await session.refresh(job)
        return user.user_id, job.processing_job_id


async def _create_manual_processing_job(
    session_factory: async_sessionmaker,
) -> tuple[int, int]:
    async with session_factory() as session:
        user = User(telegram_id=100)
        session.add(user)
        await session.flush()
        profile = await _create_profile(session, user.user_id)
        job = ProcessingJob(
            user=user,
            source_type=SourceType.MANUAL_TEXT.value,
            source_ref=None,
            raw_text="Manual English post text",
            profile_id=profile.profile_id,
            profile_snapshot=profile.profile_json,
        )
        session.add(job)
        await session.commit()
        await session.refresh(user)
        await session.refresh(job)
        return user.user_id, job.processing_job_id


async def _create_profile(session, user_id: int):
    return await ProfileService(session).upsert_profile(
        user_id,
        "B1. I want to read Reddit and ML discussions.",
        LearningProfilePayload(
            cefr_level="B1",
            level_confidence="high",
            goals_summary="Read Reddit and ML discussions.",
            focus_areas=["discussion phrases"],
            domain_interests=["Reddit", "machine learning"],
            preferred_item_types={"words": "high", "phrases": "high", "rules": "medium"},
            include=["domain vocabulary"],
            exclude=["very basic A1 words"],
            difficulty_policy="Mostly B1-B2 items.",
            extraction_guidance="Prioritize reusable discussion language.",
        ),
    )
