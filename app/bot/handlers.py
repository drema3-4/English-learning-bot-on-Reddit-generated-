from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bot.keyboards import rating_keyboard
from app.bot.messages import (
    HELP_MESSAGE,
    INVALID_RATING,
    INVALID_REDDIT_URL,
    NO_ITEMS,
    NO_JOBS,
    PROCESSING_ALREADY_ACTIVE,
    QUEUED,
    RATING_SAVED,
    REVIEW_COMPLETED,
    REVIEW_NOT_FOUND,
    REVIEW_SESSION_EXPIRED,
    SEND_REDDIT_LINK,
    START_MESSAGE,
    USER_LIMIT_REACHED,
    format_review_card,
    format_job_status,
)
from app.config import Settings
from app.services.processing_jobs import ProcessingJobService
from app.services.review import ReviewResult, ReviewService
from app.services.users import UserService
from app.utils.reddit_url import RedditUrlError


router = Router()


@router.message(CommandStart())
async def start(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    if await _ensure_user_id(message, settings, session_factory) is None:
        return
    await message.answer(START_MESSAGE)


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(HELP_MESSAGE)


@router.message(Command("status"))
async def status(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        job = await ProcessingJobService(session).get_latest_job(user_id)

    if job is None:
        await message.answer(NO_JOBS)
        return
    await message.answer(format_job_status(job.status, job.error_message))


@router.message(Command("review_words", "words"))
async def review_words(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _send_review_item(message, settings, session_factory, "words")


@router.message(Command("review_phrases", "phrases"))
async def review_phrases(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _send_review_item(message, settings, session_factory, "phrases")


@router.message(Command("review_rules", "rules"))
async def review_rules(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _send_review_item(message, settings, session_factory, "rules")


@router.message(F.text)
async def handle_text(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    text = message.text or ""
    if "reddit.com" not in text.lower():
        await message.answer(SEND_REDDIT_LINK)
        return

    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        try:
            result = await ProcessingJobService(session).queue_reddit_url(user_id, text)
        except RedditUrlError:
            await message.answer(INVALID_REDDIT_URL)
            return

    await message.answer(QUEUED if result.created else PROCESSING_ALREADY_ACTIVE)


@router.callback_query(F.data.startswith("rate:"))
async def rate_item(
    callback: CallbackQuery,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    if callback.from_user is None or callback.data is None:
        await callback.answer()
        return

    try:
        _, session_type, review_session_id, score = callback.data.split(":", maxsplit=3)
        review_session_id_int = int(review_session_id)
        score_int = int(score)
    except ValueError:
        await callback.answer(REVIEW_NOT_FOUND)
        return

    async with session_factory() as session:
        user = await UserService(session).get_by_telegram_id(callback.from_user.id)
        if user is None:
            result = ReviewResult(status="not_found")
        else:
            result = await ReviewService(session).record_rating(
                user.user_id,
                session_type,
                review_session_id_int,
                score_int,
                settings.review_session_timeout_seconds,
            )

    await _answer_rating_result(callback, result)


async def _send_review_item(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    session_type: str,
) -> None:
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        result = await ReviewService(session).start_or_continue(
            user_id,
            session_type,
            settings.review_session_timeout_seconds,
        )

    await _answer_review_result(message, result)


async def _answer_review_result(message: Message, result: ReviewResult) -> None:
    if result.status == "empty":
        await message.answer(NO_ITEMS)
        return
    if result.status == "completed":
        await message.answer(REVIEW_COMPLETED)
        return
    if result.status == "timeout":
        await message.answer(REVIEW_SESSION_EXPIRED)
        return
    if (
        result.card is None
        or result.session_type is None
        or result.review_session_id is None
    ):
        await message.answer(REVIEW_NOT_FOUND)
        return

    await message.answer(
        format_review_card(result.card),
        reply_markup=rating_keyboard(result.session_type, result.review_session_id),
    )


async def _answer_rating_result(callback: CallbackQuery, result: ReviewResult) -> None:
    if result.status == "invalid_score":
        await callback.answer(INVALID_RATING)
        return
    if result.status == "timeout":
        await callback.answer(REVIEW_SESSION_EXPIRED)
        if callback.message is not None:
            await callback.message.answer(REVIEW_SESSION_EXPIRED)
        return
    if result.status == "not_found":
        await callback.answer(REVIEW_NOT_FOUND)
        return

    await callback.answer(RATING_SAVED)
    if callback.message is None:
        return

    if result.status == "completed":
        await callback.message.answer(REVIEW_COMPLETED)
        return

    if (
        result.card is not None
        and result.session_type is not None
        and result.review_session_id is not None
    ):
        await callback.message.answer(
            format_review_card(result.card),
            reply_markup=rating_keyboard(result.session_type, result.review_session_id),
        )


async def _ensure_user_id(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> int | None:
    telegram_id = _telegram_id(message)
    username = message.from_user.username if message.from_user is not None else None
    first_name = message.from_user.first_name if message.from_user is not None else None
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(
            telegram_id,
            settings.max_users,
            username=username,
            first_name=first_name,
        )
        user_id = user.user_id if user is not None else None
    if user is None:
        await message.answer(USER_LIMIT_REACHED)
        return None
    return user_id


def _telegram_id(message: Message) -> int:
    if message.from_user is None:
        return 0
    return message.from_user.id
