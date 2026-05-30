from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.bot.handlers import handle_text, profile, profile_edit, sources, start
from app.bot.keyboards import rating_keyboard
from app.bot.messages import (
    PROFILE_EDIT_REQUEST,
    PROFILE_GENERATING,
    PROFILE_SAVED,
    PROFILE_SETUP_REQUEST,
    PROFILE_SETUP_REQUIRED,
    QUEUED_REDDIT,
    QUEUED_TEXT,
    REDDIT_SOURCE_UNAVAILABLE,
    UNKNOWN_SOURCE,
    format_review_card,
)
from app.config import Settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import (
    ProcessingJob,
    ReviewSession,
    UserBotState,
    WordLemma,
    WordSurfaceForm,
    WordUsageNote,
)
from app.services.profile_schemas import LearningProfilePayload
from app.services.profiles import AWAITING_PROFILE_INPUT, MissingLearningProfileError, ProfileService
from app.services.processing_jobs import ProcessingJobService
from app.services.review import ReviewService
from app.services.sources.types import SourceType
from app.services.users import UserService


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.from_user = SimpleNamespace(
            id=100,
            username=None,
            first_name=None,
        )
        self.answers: list[str] = []

    async def answer(self, text: str, **_: object) -> None:
        self.answers.append(text)


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


def test_settings_reddit_credentials_ignore_whitespace() -> None:
    settings = Settings(reddit_client_id="  ", reddit_client_secret="secret")

    assert settings.has_reddit_credentials is False


async def test_processing_job_service_reuses_active_job(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        await _create_profile(session, user.user_id)

        service = ProcessingJobService(session)
        first = await service.queue_reddit_post(
            user.user_id,
            "https://www.reddit.com/r/test/comments/abc123/title/",
        )
        second = await service.queue_reddit_post(
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
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/abc123/title/",
            status="processing",
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        result = await ProcessingJobService(session).queue_reddit_post(
            user.user_id,
            "https://www.reddit.com/r/test/comments/def456/other/",
        )

    assert result.created is False
    assert result.job.processing_job_id == job.processing_job_id


async def test_processing_job_service_queues_manual_text(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        await _create_profile(session, user.user_id)

        result = await ProcessingJobService(session).queue_manual_text(
            user.user_id,
            "  Manual English post text  ",
        )

    assert result.created is True
    assert result.job.source_type == SourceType.MANUAL_TEXT
    assert result.job.source_ref is None
    assert result.job.raw_text == "Manual English post text"
    assert result.job.profile_id is not None
    assert result.job.profile_snapshot is not None
    assert result.job.status == "queued"


async def test_processing_job_service_requires_profile_for_new_job(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None

        with pytest.raises(MissingLearningProfileError):
            await ProcessingJobService(session).queue_manual_text(
                user.user_id,
                "Manual English post text",
            )


async def test_processing_job_service_reuses_active_job_for_manual_text(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        await _create_profile(session, user.user_id)

        service = ProcessingJobService(session)
        first = await service.queue_manual_text(user.user_id, "Manual English post text")
        second = await service.queue_manual_text(user.user_id, "Another manual post text")

    assert first.created is True
    assert second.created is False
    assert second.job.processing_job_id == first.job.processing_job_id
    assert second.job.raw_text == "Manual English post text"


async def test_processing_job_service_creates_new_job_after_done_job(
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        await _create_profile(session, user.user_id)
        done_job = ProcessingJob(
            user_id=user.user_id,
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/abc123/title/",
            status="done",
        )
        session.add(done_job)
        await session.commit()
        await session.refresh(done_job)

        result = await ProcessingJobService(session).queue_reddit_post(
            user.user_id,
            "https://www.reddit.com/r/test/comments/def456/other/",
        )

    assert result.created is True
    assert result.job.processing_job_id != done_job.processing_job_id
    assert result.job.status == "queued"


async def test_handle_text_queues_manual_text_when_reddit_credentials_missing(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("Manual English post text")
    settings = Settings(reddit_client_id="", reddit_client_secret="", max_users=5)
    await _create_user_with_profile(session_factory)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        job = await session.scalar(select(ProcessingJob))

    assert message.answers == [QUEUED_TEXT]
    assert job is not None
    assert job.source_type == SourceType.MANUAL_TEXT
    assert job.source_ref is None
    assert job.raw_text == "Manual English post text"
    assert job.profile_id is not None
    assert job.profile_snapshot is not None


async def test_handle_text_queues_manual_text_when_reddit_credentials_exist(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("Manual English post text")
    settings = Settings(reddit_client_id="client", reddit_client_secret="secret", max_users=5)
    await _create_user_with_profile(session_factory)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        job = await session.scalar(select(ProcessingJob))

    assert message.answers == [QUEUED_TEXT]
    assert job is not None
    assert job.source_type == SourceType.MANUAL_TEXT
    assert job.raw_text == "Manual English post text"
    assert job.profile_id is not None
    assert job.profile_snapshot is not None


async def test_handle_text_queues_reddit_post_when_credentials_exist(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("https://www.reddit.com/r/test/comments/abc123/title/")
    settings = Settings(reddit_client_id="client", reddit_client_secret="secret", max_users=5)
    await _create_user_with_profile(session_factory)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        job = await session.scalar(select(ProcessingJob))

    assert message.answers == [QUEUED_REDDIT]
    assert job is not None
    assert job.source_type == SourceType.REDDIT_POST
    assert job.source_ref == "https://www.reddit.com/r/test/comments/abc123/title/"
    assert job.raw_text is None
    assert job.profile_id is not None
    assert job.profile_snapshot is not None


async def test_handle_text_rejects_reddit_post_when_credentials_missing(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("https://www.reddit.com/r/test/comments/abc123/title/")
    settings = Settings(reddit_client_id="", reddit_client_secret="", max_users=5)
    await _create_user_with_profile(session_factory)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        jobs = (await session.scalars(select(ProcessingJob))).all()

    assert message.answers == [REDDIT_SOURCE_UNAVAILABLE]
    assert jobs == []


async def test_handle_text_rejects_unknown_source_url(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("https://example.com/article")
    settings = Settings(reddit_client_id="client", reddit_client_secret="secret", max_users=5)
    await _create_user_with_profile(session_factory)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        jobs = (await session.scalars(select(ProcessingJob))).all()

    assert message.answers == [UNKNOWN_SOURCE]
    assert jobs == []


async def test_start_without_profile_requests_profile_setup(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("/start")
    settings = Settings(max_users=5)

    await start(message, settings, session_factory)

    async with session_factory() as session:
        state = await session.scalar(select(UserBotState))

    assert message.answers == [PROFILE_SETUP_REQUEST]
    assert state is not None
    assert state.state == AWAITING_PROFILE_INPUT


async def test_profile_without_profile_requests_profile_setup(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("/profile")
    settings = Settings(max_users=5)

    await profile(message, settings, session_factory)

    async with session_factory() as session:
        state = await session.scalar(select(UserBotState))

    assert message.answers == [PROFILE_SETUP_REQUEST]
    assert state is not None
    assert state.state == AWAITING_PROFILE_INPUT


async def test_profile_with_profile_shows_summary(
    session_factory: async_sessionmaker,
) -> None:
    await _create_user_with_profile(session_factory)
    message = FakeMessage("/profile")
    settings = Settings(max_users=5)

    await profile(message, settings, session_factory)

    assert len(message.answers) == 1
    assert "Твой учебный профиль" in message.answers[0]
    assert "Уровень: B1" in message.answers[0]
    assert "Reddit" in message.answers[0]


async def test_profile_edit_sets_awaiting_state(
    session_factory: async_sessionmaker,
) -> None:
    await _create_user_with_profile(session_factory)
    message = FakeMessage("/profile_edit")
    settings = Settings(max_users=5)

    await profile_edit(message, settings, session_factory)

    async with session_factory() as session:
        state = await session.scalar(select(UserBotState))

    assert message.answers == [PROFILE_EDIT_REQUEST]
    assert state is not None
    assert state.state == AWAITING_PROFILE_INPUT


async def test_handle_text_without_profile_requests_setup_and_creates_no_job(
    session_factory: async_sessionmaker,
) -> None:
    message = FakeMessage("Manual English post text")
    settings = Settings(max_users=5)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        jobs = (await session.scalars(select(ProcessingJob))).all()
        state = await session.scalar(select(UserBotState))

    assert message.answers == [PROFILE_SETUP_REQUIRED]
    assert jobs == []
    assert state is not None
    assert state.state == AWAITING_PROFILE_INPUT


async def test_handle_text_awaiting_profile_generates_profile(
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        await ProfileService(session).set_awaiting_profile_input(user.user_id, reason="start")

    class FakeProfileGenerationService:
        def __init__(self, session, *_: object, **__: object) -> None:
            self.session = session

        async def generate_profile(self, user_id: int, raw_user_input: str):
            return await ProfileService(self.session).upsert_profile(
                user_id,
                raw_user_input,
                _profile_payload(),
            )

    monkeypatch.setattr(
        "app.bot.handlers.ProfileGenerationService",
        FakeProfileGenerationService,
    )
    message = FakeMessage("B1. I want Reddit and ML vocabulary.")
    settings = Settings(max_users=5)

    await handle_text(message, settings, session_factory)

    async with session_factory() as session:
        jobs = (await session.scalars(select(ProcessingJob))).all()
        state = await session.scalar(select(UserBotState))
        profile_row = await ProfileService(session).get_active_profile(1)

    assert message.answers[0] == PROFILE_GENERATING
    assert PROFILE_SAVED in message.answers[1]
    assert jobs == []
    assert state is None
    assert profile_row is not None


async def test_sources_command_lists_configured_sources() -> None:
    message = FakeMessage("/sources")
    settings = Settings(reddit_client_id="", reddit_client_secret="", max_users=5)

    await sources(message, settings)

    assert "Текст вручную: настроено" in message.answers[0]
    assert "Reddit: не настроено" in message.answers[0]


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


async def _create_user_with_profile(session_factory: async_sessionmaker) -> int:
    async with session_factory() as session:
        user = await UserService(session).ensure_allowed(telegram_id=100, max_users=5)
        assert user is not None
        await _create_profile(session, user.user_id)
        return user.user_id


async def _create_profile(session, user_id: int):
    return await ProfileService(session).upsert_profile(
        user_id,
        "B1. I want to read Reddit and ML discussions.",
        _profile_payload(),
    )


def _profile_payload() -> LearningProfilePayload:
    return LearningProfilePayload(
        cefr_level="B1",
        level_confidence="high",
        goals_summary="Read Reddit and machine learning discussions.",
        focus_areas=["phrasal verbs", "discussion phrases"],
        domain_interests=["Reddit", "machine learning"],
        preferred_item_types={"words": "high", "phrases": "high", "rules": "medium"},
        include=["domain vocabulary"],
        exclude=["very basic A1 words"],
        difficulty_policy="Mostly B1-B2 practical items.",
        extraction_guidance="Prioritize reusable discussion language.",
    )
