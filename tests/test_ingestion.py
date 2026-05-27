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


class FakeRedditService:
    def __init__(self, text: str = "Reddit source text") -> None:
        self.text = text
        self.calls: list[tuple[str, int]] = []

    async def fetch_post_text(self, url: str, comments_limit: int = 20) -> str:
        self.calls.append((url, comments_limit))
        return self.text


class FakeExtractionService:
    async def extract_words(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[WordExtract]:
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
    ) -> list[PhraseExtract]:
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
    ) -> list[RuleExtract]:
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
    service = IngestionService(
        session_factory,
        reddit_service,
        FakeExtractionService(),
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
    assert [lemma.lemma for lemma in word_lemmas] == ["notice"]
    assert len(word_surface_forms) == 1
    assert len(word_usage_notes) == 1
    assert len(phrases) == 1
    assert len(phrase_functions) == 1
    assert len(phrase_examples) == 1
    assert len(rules) == 1
    assert len(rule_examples) == 1
    assert user_id == word_lemmas[0].user_id


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
        job = ProcessingJob(
            user=user,
            reddit_url="https://www.reddit.com/r/test/comments/abc123/title/",
        )
        session.add(job)
        await session.commit()
        await session.refresh(user)
        await session.refresh(job)
        return user.user_id, job.processing_job_id
