from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import User, UserBotState, UserLearningProfile
from app.services.profile_prompts import render_profile_for_prompt
from app.services.profile_schemas import LearningProfilePayload
from app.services.profiles import AWAITING_PROFILE_INPUT, ProfileService


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


async def test_upsert_profile_creates_profile(session_factory: async_sessionmaker) -> None:
    user_id = await _create_user(session_factory)

    async with session_factory() as session:
        profile = await ProfileService(session).upsert_profile(
            user_id,
            "B1. I want Reddit vocabulary.",
            _profile_payload(),
        )

    assert profile.profile_id is not None
    assert profile.cefr_level == "B1"
    assert "Reddit" in profile.goals_summary


async def test_upsert_profile_updates_existing_profile(
    session_factory: async_sessionmaker,
) -> None:
    user_id = await _create_user(session_factory)

    async with session_factory() as session:
        service = ProfileService(session)
        first = await service.upsert_profile(
            user_id,
            "B1. I want Reddit vocabulary.",
            _profile_payload(),
        )
        second = await service.upsert_profile(
            user_id,
            "C1. I want idioms.",
            _profile_payload(cefr_level="C1", goals_summary="Understand idioms."),
        )
        profiles = (await session.scalars(select(UserLearningProfile))).all()

    assert second.profile_id == first.profile_id
    assert second.cefr_level == "C1"
    assert len(profiles) == 1


async def test_state_is_created_updated_and_cleared(
    session_factory: async_sessionmaker,
) -> None:
    user_id = await _create_user(session_factory)

    async with session_factory() as session:
        service = ProfileService(session)
        await service.set_awaiting_profile_input(user_id, reason="start")
        await service.set_awaiting_profile_input(user_id, reason="edit")
        states = (await session.scalars(select(UserBotState))).all()
        state = await service.get_state(user_id)

        await service.clear_state(user_id)
        cleared = await service.get_state(user_id)

    assert len(states) == 1
    assert state is not None
    assert state.state == AWAITING_PROFILE_INPUT
    assert '"edit"' in (state.payload or "")
    assert cleared is None


def test_render_profile_for_prompt_does_not_include_raw_input() -> None:
    payload = _profile_payload()

    rendered = render_profile_for_prompt(payload)

    assert "User learning profile" in rendered
    assert "B1. I want Reddit vocabulary." not in rendered
    assert "English level: B1" in rendered


async def _create_user(session_factory: async_sessionmaker) -> int:
    async with session_factory() as session:
        user = User(telegram_id=100)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.user_id


def _profile_payload(
    cefr_level: str = "B1",
    goals_summary: str = "Read Reddit and machine learning discussions.",
) -> LearningProfilePayload:
    return LearningProfilePayload(
        cefr_level=cefr_level,
        level_confidence="high",
        goals_summary=goals_summary,
        focus_areas=["phrasal verbs", "discussion phrases"],
        domain_interests=["Reddit", "machine learning"],
        preferred_item_types={"words": "high", "phrases": "high", "rules": "medium"},
        include=["domain vocabulary"],
        exclude=["very basic A1 words"],
        difficulty_policy="Mostly B1-B2 practical items.",
        extraction_guidance="Prioritize reusable discussion language.",
    )
