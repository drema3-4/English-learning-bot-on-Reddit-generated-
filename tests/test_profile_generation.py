from __future__ import annotations

import json

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import LLMProfileJob, User
from app.services.profile_generation import ProfileGenerationService
from app.services.profiles import ProfileGenerationError, ProfileService


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def complete_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


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


async def test_generate_profile_creates_profile_from_valid_json(
    session_factory: async_sessionmaker,
) -> None:
    user_id = await _create_user(session_factory)
    fake_llm = FakeLLM(json.dumps(_profile_payload()))

    async with session_factory() as session:
        profile = await ProfileGenerationService(session, fake_llm).generate_profile(
            user_id,
            "B1. I want Reddit and ML vocabulary.",
        )
        job = await session.scalar(select(LLMProfileJob))

    assert profile.cefr_level == "B1"
    assert "User learning profile" in profile.prompt_profile
    assert job is not None
    assert job.status == "done"
    assert job.raw_response == fake_llm.response


async def test_generate_profile_parses_json_inside_markdown(
    session_factory: async_sessionmaker,
) -> None:
    user_id = await _create_user(session_factory)
    fake_llm = FakeLLM(f"```json\n{json.dumps(_profile_payload())}\n```")

    async with session_factory() as session:
        profile = await ProfileGenerationService(session, fake_llm).generate_profile(
            user_id,
            "B1. I want Reddit and ML vocabulary.",
        )

    assert profile.cefr_level == "B1"


async def test_generate_profile_marks_job_failed_on_invalid_json(
    session_factory: async_sessionmaker,
) -> None:
    user_id = await _create_user(session_factory)
    fake_llm = FakeLLM("not json")

    async with session_factory() as session:
        with pytest.raises(ProfileGenerationError):
            await ProfileGenerationService(session, fake_llm).generate_profile(
                user_id,
                "B1. I want Reddit and ML vocabulary.",
            )
        job = await session.scalar(select(LLMProfileJob))

    assert job is not None
    assert job.status == "failed"
    assert job.raw_response == "not json"


async def test_failed_generation_keeps_existing_profile(
    session_factory: async_sessionmaker,
) -> None:
    user_id = await _create_user(session_factory)

    async with session_factory() as session:
        await ProfileGenerationService(
            session,
            FakeLLM(json.dumps(_profile_payload())),
        ).generate_profile(user_id, "B1. I want Reddit vocabulary.")

    async with session_factory() as session:
        with pytest.raises(ProfileGenerationError):
            await ProfileGenerationService(session, FakeLLM("not json")).generate_profile(
                user_id,
                "C1. I want idioms.",
            )
        active_profile = await ProfileService(session).get_active_profile(user_id)

    assert active_profile is not None
    assert active_profile.cefr_level == "B1"


async def test_missing_level_defaults_to_unknown(session_factory: async_sessionmaker) -> None:
    user_id = await _create_user(session_factory)
    payload = _profile_payload()
    payload.pop("cefr_level")
    fake_llm = FakeLLM(json.dumps(payload))

    async with session_factory() as session:
        profile = await ProfileGenerationService(session, fake_llm).generate_profile(
            user_id,
            "I want to understand Reddit.",
        )

    assert profile.cefr_level == "unknown"


async def _create_user(session_factory: async_sessionmaker) -> int:
    async with session_factory() as session:
        user = User(telegram_id=100)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.user_id


def _profile_payload() -> dict[str, object]:
    return {
        "cefr_level": "B1",
        "level_confidence": "high",
        "goals_summary": "Read Reddit and machine learning discussions.",
        "focus_areas": ["phrasal verbs", "discussion phrases"],
        "domain_interests": ["Reddit", "machine learning"],
        "preferred_item_types": {"words": "high", "phrases": "high", "rules": "medium"},
        "include": ["domain vocabulary"],
        "exclude": ["very basic A1 words"],
        "difficulty_policy": "Mostly B1-B2 practical items.",
        "extraction_guidance": "Prioritize reusable discussion language.",
    }
