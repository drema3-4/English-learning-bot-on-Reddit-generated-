from __future__ import annotations

import json

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import LLMExtractionJob, ProcessingJob, User
from app.services.extraction import (
    MAX_WORDS,
    ExtractionService,
    PhraseExtract,
    RuleExtract,
    WordExtract,
    build_learning_items,
    normalize_llm_payload,
)
from app.utils.json_parse import JSONParseError


class FakeLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def complete_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


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


def test_normalize_llm_payload_collects_supported_items() -> None:
    payload = {
        "words": [{"text": "quirky", "translation": "strange"}],
        "phrases": [{"text": "show up", "example": "He did not show up."}],
        "rules": [{"title": "Present perfect", "explanation": "Past action with present result."}],
    }

    items = normalize_llm_payload(payload)

    assert [item.item_type for item in items] == ["word", "phrase", "rule"]
    assert [item.text for item in items] == ["quirky", "show up", "Present perfect"]


def test_normalize_llm_payload_deduplicates_case_insensitively() -> None:
    payload = {
        "words": [
            {"text": "Awkward"},
            {"text": "awkward"},
        ]
    }

    items = normalize_llm_payload(payload)

    assert len(items) == 1


def test_build_learning_items_preserves_structured_fields() -> None:
    word = WordExtract(
        lemma="show up",
        surface_form="showed up",
        meaning_en="appeared",
        meaning_ru="pojavilsa",
        usage_note="He showed up late.",
        usage_note_translation="On pojavilsa pozdno.",
    )
    phrase = PhraseExtract(
        phrase="to be fair",
        function="softens disagreement",
        meaning_en="adds balance before an argument",
        meaning_ru="spravedlivosti radi",
        example="To be fair, the model is small.",
        example_translation="Spravedlivosti radi, model malen'kaya.",
    )
    rule = RuleExtract(
        rule_en="Use would to soften opinions.",
        rule_ru="Would smyagchaet mnenie.",
        example="I would say this is overfitting.",
        example_translation="Ya by skazal, chto eto pereobuchenie.",
    )

    items = build_learning_items([word], [phrase], [rule])

    assert [item.item_type for item in items] == ["word", "phrase", "rule"]
    assert items[0].lemma == "show up"
    assert items[0].example_translation == "On pojavilsa pozdno."
    assert items[1].function == "softens disagreement"
    assert items[2].rule_ru == "Would smyagchaet mnenie."


async def test_extract_words_creates_done_job_and_limits_results(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    payload = [
        {
            "lemma": f"term{i}",
            "surface_form": f"term{i}",
            "meaning_en": "technical term",
            "meaning_ru": "termin",
            "usage_note": f"Term {i} appears in the thread.",
            "usage_note_translation": f"Termin {i} est v obsuzhdenii.",
        }
        for i in range(MAX_WORDS + 1)
    ]
    raw_response = json.dumps(payload)
    fake_llm = FakeLLM(raw_response)

    result = await ExtractionService(session_factory, fake_llm).extract_words(
        user_id,
        processing_job_id,
        "This ML discussion mentions model alignment.",
    )

    assert len(result) == MAX_WORDS
    assert result[0].lemma == "term0"
    assert "do not extract articles" in fake_llm.prompts[0]
    assert "ML/DL/NLP" in fake_llm.prompts[0]

    async with session_factory() as session:
        jobs = (await session.scalars(select(LLMExtractionJob))).all()

    assert len(jobs) == 1
    assert jobs[0].job_type == "words"
    assert jobs[0].status == "done"
    assert jobs[0].input_text == "This ML discussion mentions model alignment."
    assert jobs[0].prompt_text == fake_llm.prompts[0]
    assert jobs[0].raw_response == raw_response
    assert len(json.loads(jobs[0].parsed_response or "[]")) == MAX_WORDS
    assert jobs[0].error_message is None


async def test_extract_rules_marks_failed_job_on_bad_json(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    fake_llm = FakeLLM("not json")

    with pytest.raises(JSONParseError):
        await ExtractionService(session_factory, fake_llm).extract_rules(
            user_id,
            processing_job_id,
            "A sentence without parseable output.",
        )

    async with session_factory() as session:
        job = await session.scalar(select(LLMExtractionJob))

    assert job is not None
    assert job.job_type == "rules"
    assert job.status == "failed"
    assert job.raw_response == "not json"
    assert job.error_message is not None
    assert "Could not parse a JSON array" in job.error_message


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
