from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.bot.keyboards import rating_keyboard
from app.bot.messages import format_review_card
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import ProcessingJob, ReviewSession, WordLemma, WordSurfaceForm, WordUsageNote
from app.services.processing_jobs import ProcessingJobService
from app.services.review import ReviewService
from app.services.users import UserService


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


async def test_user_service_limits_new_users(session_factory: async_sessionmaker) -> None:
    async with session_factory() as session:
        service = UserService(session)

        created_users = [
            await service.ensure_allowed(telegram_id=telegram_id, max_users=5)
            for telegram_id in range(100, 105)
        ]
        sixth_user = await service.ensure_allowed(telegram_id=105, max_users=5)
        existing_user = await service.ensure_allowed(telegram_id=102, max_users=5)

    assert all(user is not None for user in created_users)
    assert sixth_user is None
    assert existing_user is not None
    assert existing_user.user_id == created_users[2].user_id


async def test_processing_job_service_reuses_active_job(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None

        service = ProcessingJobService(session)
        first = await service.queue_reddit_url(
            user.user_id,
            "https://www.reddit.com/r/test/comments/abc123/title/",
        )
        second = await service.queue_reddit_url(
            user.user_id,
            "https://www.reddit.com/r/test/comments/def456/other/",
        )

    assert first.created is True
    assert second.created is False
    assert second.job.processing_job_id == first.job.processing_job_id


async def test_processing_job_service_reuses_processing_job(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        job = ProcessingJob(
            user_id=user.user_id,
            reddit_url="https://www.reddit.com/r/test/comments/abc123/title/",
            status="processing",
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        result = await ProcessingJobService(session).queue_reddit_url(
            user.user_id,
            "https://www.reddit.com/r/test/comments/def456/other/",
        )

    assert result.created is False
    assert result.job.processing_job_id == job.processing_job_id


async def test_processing_job_service_creates_new_job_after_done_job(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        done_job = ProcessingJob(
            user_id=user.user_id,
            reddit_url="https://www.reddit.com/r/test/comments/abc123/title/",
            status="done",
        )
        session.add(done_job)
        await session.commit()
        await session.refresh(done_job)

        result = await ProcessingJobService(session).queue_reddit_url(
            user.user_id,
            "https://www.reddit.com/r/test/comments/def456/other/",
        )

    assert result.created is True
    assert result.job.processing_job_id != done_job.processing_job_id
    assert result.job.status == "queued"


async def test_review_service_records_word_rating_and_completes_session(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None

        lemma = WordLemma(lemma="notice", user_id=user.user_id)
        surface_form = WordSurfaceForm(
            surface_form="noticed",
            meaning_en="became aware of",
            meaning_ru="заметил",
            lemma=lemma,
        )
        usage_note = WordUsageNote(
            usage_note="I noticed the pattern.",
            usage_note_translation="Я заметил закономерность.",
            surface_form=surface_form,
        )
        session.add(usage_note)
        await session.commit()

        service = ReviewService(session)
        first_card = await service.start_or_continue(user.user_id, "words", timeout_seconds=120)
        assert first_card.status == "card"
        assert first_card.card is not None
        assert first_card.review_session_id is not None
        assert "Слово: noticed" in format_review_card(first_card.card)

        keyboard = rating_keyboard("words", first_card.review_session_id)
        assert keyboard.inline_keyboard[0][4].callback_data == (
            f"rate:words:{first_card.review_session_id}:5"
        )

        result = await service.record_rating(
            user.user_id,
            "words",
            first_card.review_session_id,
            4,
            timeout_seconds=120,
        )

        review_session = await session.get(ReviewSession, first_card.review_session_id)
        assert review_session is not None

    assert result.status == "completed"
    assert lemma.current_score == 4
    assert surface_form.current_score == 4
    assert usage_note.current_score == 4
    assert usage_note.last_repetition is not None
    assert review_session.status == "finished"


async def test_review_service_times_out_stale_session(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None

        stale_session = ReviewSession(
            user_id=user.user_id,
            session_type="words",
            items="[]",
            status="active",
            updated_at=datetime.now(UTC) - timedelta(seconds=300),
        )
        session.add(stale_session)
        await session.commit()

        result = await ReviewService(session).start_or_continue(
            user.user_id,
            "words",
            timeout_seconds=120,
        )

    assert result.status == "empty"
    assert stale_session.status == "timeout"
