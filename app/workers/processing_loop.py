from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from aiogram import Bot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings, get_settings
from app.db.models import ProcessingJob, User
from app.db.session import create_session_factory
from app.services.extraction import ExtractionService
from app.services.ingestion import IngestionService
from app.services.llm import LLMClient
from app.services.reddit import RedditService


POLL_INTERVAL_SECONDS = 3
SUCCESS_MESSAGE = """Обработка завершена.

Добавлены/обновлены:
— слова;
— фразы;
— правила.

Теперь можно запустить:
/review_words
/review_phrases
/review_rules"""
ERROR_MESSAGE_TEMPLATE = """Не удалось обработать материал.

Причина:
{error_message}"""

logger = logging.getLogger(__name__)


async def processing_loop(
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    bot: Bot | None = None,
) -> None:
    settings = settings or get_settings()
    session_factory = session_factory or create_session_factory(settings)

    reddit_service = RedditService(settings)
    llm_client = LLMClient(settings)
    extraction_service = ExtractionService(session_factory, llm_client)
    ingestion_service = IngestionService(
        session_factory,
        reddit_service,
        extraction_service,
        comments_limit=settings.reddit_comments_limit,
    )

    while True:
        await mark_timed_out_jobs(settings, session_factory)
        await process_next_job(
            settings,
            session_factory,
            reddit_service=reddit_service,
            extraction_service=extraction_service,
            ingestion_service=ingestion_service,
            bot=bot,
        )
        await mark_timed_out_jobs(settings, session_factory)
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def process_next_job(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    reddit_service: RedditService | None = None,
    extraction_service: ExtractionService | None = None,
    bot: Bot | None = None,
    ingestion_service: IngestionService | None = None,
) -> bool:
    claimed = await _claim_next_job(session_factory)
    if claimed is None:
        return False

    job_id, telegram_id = claimed
    if ingestion_service is None:
        reddit_service = reddit_service or RedditService(settings)
        if extraction_service is None:
            extraction_service = ExtractionService(session_factory, LLMClient(settings))
        ingestion_service = IngestionService(
            session_factory,
            reddit_service,
            extraction_service,
            comments_limit=settings.reddit_comments_limit,
        )

    try:
        await ingestion_service.process_job(job_id)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        await _mark_failed(session_factory, job_id, error_message)
        await _notify_failure(bot, telegram_id, error_message)
        return True

    await _mark_done(session_factory, job_id)
    await _notify_success(bot, telegram_id)
    return True


async def mark_timed_out_jobs(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> int:
    now = datetime.now(UTC)
    started_before = now - timedelta(seconds=settings.processing_job_timeout_seconds)
    async with session_factory() as session:
        jobs = (
            await session.scalars(
                select(ProcessingJob).where(
                    ProcessingJob.status == "processing",
                    ProcessingJob.started_at.is_not(None),
                    ProcessingJob.started_at < started_before,
                )
            )
        ).all()
        for job in jobs:
            job.status = "timeout"
            job.error_message = "Processing job timed out."
            job.finished_at = now

        if jobs:
            await session.commit()
        return len(jobs)


async def _claim_next_job(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[int, int] | None:
    async with session_factory() as session:
        result = await session.execute(
            select(ProcessingJob, User.telegram_id)
            .join(User, ProcessingJob.user_id == User.user_id)
            .where(ProcessingJob.status == "queued")
            .order_by(ProcessingJob.created_at.asc(), ProcessingJob.processing_job_id.asc())
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None

        job, telegram_id = row
        job.status = "processing"
        job.started_at = datetime.now(UTC)
        job.error_message = None
        await session.commit()
        return job.processing_job_id, telegram_id


async def _mark_done(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: int,
) -> None:
    async with session_factory() as session:
        job = await session.get(ProcessingJob, job_id)
        if job is None:
            return

        job.status = "done"
        job.error_message = None
        job.finished_at = datetime.now(UTC)
        await session.commit()


async def _mark_failed(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: int,
    error_message: str,
) -> None:
    async with session_factory() as session:
        job = await session.get(ProcessingJob, job_id)
        if job is None:
            return

        job.status = "failed"
        job.error_message = error_message[:4000]
        job.finished_at = datetime.now(UTC)
        await session.commit()


async def _notify_success(bot: Bot | None, telegram_id: int) -> None:
    if bot is None:
        return
    try:
        await bot.send_message(telegram_id, SUCCESS_MESSAGE)
    except Exception:  # noqa: BLE001
        logger.exception("Could not send processing success notification")


async def _notify_failure(bot: Bot | None, telegram_id: int, error_message: str) -> None:
    if bot is None:
        return
    try:
        await bot.send_message(
            telegram_id,
            ERROR_MESSAGE_TEMPLATE.format(error_message=_public_error_message(error_message)),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Could not send processing failure notification")


def _public_error_message(error_message: str) -> str:
    if "Could not parse a JSON" in error_message or "Empty JSON payload" in error_message:
        return (
            "LLM вернула ответ не в ожидаемом JSON-формате. "
            "Попробуй отправить текст ещё раз или выбрать другую модель."
        )
    return error_message
