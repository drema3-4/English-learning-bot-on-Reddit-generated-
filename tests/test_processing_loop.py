from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import ProcessingJob, User
from app.services.sources.types import SourceType
from app.workers.processing_loop import mark_timed_out_jobs, process_next_job


class FakeIngestionService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.processed_job_ids: list[int] = []

    async def process_job(self, job_id: int) -> None:
        self.processed_job_ids.append(job_id)
        if self.error is not None:
            raise self.error


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, telegram_id: int, text: str) -> None:
        self.messages.append((telegram_id, text))


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


async def test_process_next_job_claims_oldest_job_and_marks_done(
    session_factory: async_sessionmaker,
) -> None:
    first_job_id, second_job_id = await _create_two_queued_jobs(session_factory)
    ingestion_service = FakeIngestionService()
    bot = FakeBot()

    processed = await process_next_job(
        _settings(),
        session_factory,
        bot=bot,
        ingestion_service=ingestion_service,
    )

    async with session_factory() as session:
        first_job = await session.get(ProcessingJob, first_job_id)
        second_job = await session.get(ProcessingJob, second_job_id)

    assert processed is True
    assert ingestion_service.processed_job_ids == [first_job_id]
    assert first_job is not None
    assert first_job.status == "done"
    assert first_job.started_at is not None
    assert first_job.finished_at is not None
    assert second_job is not None
    assert second_job.status == "queued"
    assert bot.messages[0][0] == 100
    assert "Обработка завершена." in bot.messages[0][1]


async def test_process_next_job_marks_failed_on_ingestion_error(
    session_factory: async_sessionmaker,
) -> None:
    first_job_id, _ = await _create_two_queued_jobs(session_factory)
    ingestion_service = FakeIngestionService(RuntimeError("boom"))
    bot = FakeBot()

    processed = await process_next_job(
        _settings(),
        session_factory,
        bot=bot,
        ingestion_service=ingestion_service,
    )

    async with session_factory() as session:
        job = await session.get(ProcessingJob, first_job_id)

    assert processed is True
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "boom"
    assert job.finished_at is not None
    assert "Причина:\nboom" in bot.messages[0][1]


async def test_process_next_job_sends_generic_source_failure_message(
    session_factory: async_sessionmaker,
) -> None:
    first_job_id, _ = await _create_two_queued_jobs(session_factory)
    ingestion_service = FakeIngestionService(RuntimeError("Could not parse a JSON array"))
    bot = FakeBot()

    processed = await process_next_job(
        _settings(),
        session_factory,
        bot=bot,
        ingestion_service=ingestion_service,
    )

    async with session_factory() as session:
        job = await session.get(ProcessingJob, first_job_id)

    assert processed is True
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "Could not parse a JSON array"
    assert "Не удалось обработать материал." in bot.messages[0][1]
    assert "LLM вернула ответ не в ожидаемом JSON-формате." in bot.messages[0][1]


async def test_mark_timed_out_jobs_marks_stale_processing_jobs(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = User(telegram_id=100)
        job = ProcessingJob(
            user=user,
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/abc123/title/",
            status="processing",
            started_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.processing_job_id

    timed_out = await mark_timed_out_jobs(_settings(timeout_seconds=1), session_factory)

    async with session_factory() as session:
        job = await session.get(ProcessingJob, job_id)

    assert timed_out == 1
    assert job is not None
    assert job.status == "timeout"
    assert job.error_message == "Processing job timed out."
    assert job.finished_at is not None


async def _create_two_queued_jobs(
    session_factory: async_sessionmaker,
) -> tuple[int, int]:
    async with session_factory() as session:
        user = User(telegram_id=100)
        first_job = ProcessingJob(
            user=user,
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/abc123/title/",
        )
        second_job = ProcessingJob(
            user=user,
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/def456/title/",
        )
        session.add_all([first_job, second_job])
        await session.commit()
        await session.refresh(first_job)
        await session.refresh(second_job)
        return first_job.processing_job_id, second_job.processing_job_id


def _settings(timeout_seconds: int = 120) -> Settings:
    return Settings(
        telegram_bot_token="",
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4.1-mini",
        reddit_client_id="",
        reddit_client_secret="",
        reddit_user_agent="test",
        database_url="sqlite+aiosqlite:///:memory:",
        max_users=5,
        reddit_comments_limit=20,
        processing_job_timeout_seconds=timeout_seconds,
        review_session_timeout_seconds=120,
    )
