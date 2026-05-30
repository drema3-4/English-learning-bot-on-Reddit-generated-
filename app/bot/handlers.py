from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bot.keyboards import rating_keyboard
from app.bot.messages import (
    INVALID_RATING,
    NO_ITEMS,
    NO_JOBS,
    PROCESSING_ALREADY_ACTIVE,
    PROFILE_CANCELLED,
    PROFILE_EDIT_REQUEST,
    PROFILE_GENERATING,
    PROFILE_GENERATION_FAILED,
    PROFILE_SAVED,
    PROFILE_SETUP_REQUEST,
    PROFILE_SETUP_REQUIRED,
    QUEUED_REDDIT,
    QUEUED_TEXT,
    RATING_SAVED,
    SEND_SOURCE_INPUT,
    REVIEW_COMPLETED,
    REVIEW_NOT_FOUND,
    REVIEW_SESSION_EXPIRED,
    UNKNOWN_SOURCE,
    USER_LIMIT_REACHED,
    format_job_status,
    format_profile,
    format_review_card,
    format_sources_status,
    help_message,
    start_message,
)
from app.config import Settings
from app.services.llm import LLMClient
from app.services.profile_generation import ProfileGenerationService
from app.services.processing_jobs import ManualPostTextError, ProcessingJobService
from app.services.profiles import (
    AWAITING_PROFILE_INPUT,
    MissingLearningProfileError,
    ProfileGenerationError,
    ProfileService,
)
from app.services.review import ReviewResult, ReviewService
from app.services.sources import (
    SourceAvailabilityService,
    SourceDetectionStatus,
    SourceDetector,
    SourceType,
)
from app.services.users import UserService


router = Router()


@router.message(CommandStart())
async def start(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return
    async with session_factory() as session:
        profile_service = ProfileService(session)
        profile = await profile_service.get_active_profile(user_id)
        if profile is None and settings.profile_required:
            await profile_service.set_awaiting_profile_input(user_id, reason="start")
            await message.answer(PROFILE_SETUP_REQUEST)
            return
    await message.answer(start_message(settings.has_reddit_credentials))


@router.message(Command("help"))
async def help_command(message: Message, settings: Settings) -> None:
    await message.answer(help_message(settings.has_reddit_credentials))


@router.message(Command("profile"))
async def profile(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        profile_service = ProfileService(session)
        active_profile = await profile_service.get_active_profile(user_id)
        if active_profile is None:
            await profile_service.set_awaiting_profile_input(user_id, reason="profile")
            await message.answer(PROFILE_SETUP_REQUEST)
            return
        await message.answer(format_profile(active_profile))


@router.message(Command("profile_edit"))
async def profile_edit(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        await ProfileService(session).set_awaiting_profile_input(user_id, reason="edit")
    await message.answer(PROFILE_EDIT_REQUEST)


@router.message(Command("profile_cancel"))
async def profile_cancel(
    message: Message,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        await ProfileService(session).clear_state(user_id)
    await message.answer(PROFILE_CANCELLED)


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
    await message.answer(format_job_status(job.status, job.error_message, job.source_type))


@router.message(Command("sources"))
async def sources(message: Message, settings: Settings) -> None:
    service = SourceAvailabilityService(settings)
    await message.answer(format_sources_status(service.list_sources()))


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
    user_id = await _ensure_user_id(message, settings, session_factory)
    if user_id is None:
        return

    async with session_factory() as session:
        profile_service = ProfileService(session)
        profile_state = await profile_service.get_state(user_id)
        if profile_state is not None and profile_state.state == AWAITING_PROFILE_INPUT:
            await _handle_profile_input(message, settings, session, user_id)
            return

        active_profile = await profile_service.get_active_profile(user_id)
        if active_profile is None and settings.profile_required:
            await profile_service.set_awaiting_profile_input(user_id, reason="missing")
            await message.answer(PROFILE_SETUP_REQUIRED)
            return

    detected_source = SourceDetector().detect(message.text)
    if detected_source.status == SourceDetectionStatus.EMPTY_INPUT:
        await message.answer(SEND_SOURCE_INPUT)
        return
    if detected_source.status == SourceDetectionStatus.UNKNOWN_SOURCE:
        await message.answer(UNKNOWN_SOURCE)
        return
    if detected_source.source_type is None:
        await message.answer(UNKNOWN_SOURCE)
        return

    availability_service = SourceAvailabilityService(settings)
    source_availability = availability_service.get_source(detected_source.source_type)
    if not source_availability.is_configured:
        await message.answer(source_availability.unavailable_message)
        return

    async with session_factory() as session:
        service = ProcessingJobService(session)
        try:
            result = await service.queue_source(
                user_id=user_id,
                source_type=detected_source.source_type,
                source_ref=detected_source.source_ref,
                raw_text=detected_source.input_text,
                require_profile=settings.profile_required,
            )
        except ManualPostTextError:
            await message.answer(SEND_SOURCE_INPUT)
            return
        except MissingLearningProfileError:
            await ProfileService(session).set_awaiting_profile_input(
                user_id,
                reason="missing",
            )
            await message.answer(PROFILE_SETUP_REQUIRED)
            return

    if not result.created:
        await message.answer(PROCESSING_ALREADY_ACTIVE)
        return
    if detected_source.source_type == SourceType.REDDIT_POST:
        await message.answer(QUEUED_REDDIT)
        return
    await message.answer(QUEUED_TEXT)


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


async def _handle_profile_input(
    message: Message,
    settings: Settings,
    session: AsyncSession,
    user_id: int,
) -> None:
    raw_text = (message.text or "").strip()
    if not raw_text:
        await message.answer(PROFILE_SETUP_REQUIRED)
        return

    await message.answer(PROFILE_GENERATING)
    try:
        profile = await ProfileGenerationService(
            session,
            LLMClient(settings),
            max_input_chars=settings.profile_generation_max_input_chars,
        ).generate_profile(user_id, raw_text)
    except ProfileGenerationError:
        await message.answer(PROFILE_GENERATION_FAILED)
        return

    await ProfileService(session).clear_state(user_id)
    await message.answer(f"{PROFILE_SAVED}\n\n{format_profile(profile)}")


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
