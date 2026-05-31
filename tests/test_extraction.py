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
from app.services.profile_schemas import LearningProfilePayload
from app.services.profiles import ProfileService
from app.services.sources.types import SourceType
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


async def test_extract_words_accepts_wrapped_payload(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    raw_response = json.dumps(
        {
            "words": [
                {
                    "lemma": "filtering",
                    "surface_form": "filtering",
                    "meaning_en": "removing less useful features",
                    "meaning_ru": "фильтрация",
                    "usage_note": "What type of filtering should I do?",
                    "usage_note_translation": "Какую фильтрацию мне делать?",
                }
            ]
        }
    )
    fake_llm = FakeLLM(raw_response)

    result = await ExtractionService(session_factory, fake_llm).extract_words(
        user_id,
        processing_job_id,
        "What type of filtering should I do for xgboost?",
    )

    assert [item.lemma for item in result] == ["filtering"]


async def test_extract_words_includes_profile_prompt_and_logs_snapshot(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    profile_id, profile_snapshot = await _get_profile_snapshot(session_factory, user_id)
    raw_response = json.dumps(
        [
            {
                "lemma": "alignment",
                "surface_form": "alignment",
                "meaning_en": "making model behavior match goals",
                "meaning_ru": "согласование",
                "usage_note": "Model alignment matters.",
                "usage_note_translation": "Согласование модели важно.",
            }
        ]
    )
    fake_llm = FakeLLM(raw_response)

    await ExtractionService(session_factory, fake_llm).extract_words(
        user_id,
        processing_job_id,
        "Model alignment matters.",
        profile_id=profile_id,
        profile_snapshot=profile_snapshot,
    )

    async with session_factory() as session:
        job = await session.scalar(select(LLMExtractionJob))

    assert "User learning profile" in fake_llm.prompts[0]
    assert "English level: B1" in fake_llm.prompts[0]
    assert "Read Reddit and machine learning discussions" in fake_llm.prompts[0]
    assert job is not None
    assert job.prompt_text == fake_llm.prompts[0]
    assert job.profile_id == profile_id
    assert job.profile_snapshot == profile_snapshot


async def test_chunked_extraction_deduplicates_and_limits_after_chunks(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    profile_id, profile_snapshot = await _get_profile_snapshot(session_factory, user_id)
    first_response = json.dumps(
        [
            {
                "lemma": "alignment",
                "surface_form": "alignment",
                "meaning_en": "matching goals",
                "meaning_ru": "согласование",
                "usage_note": "Alignment matters.",
                "usage_note_translation": "Согласование важно.",
            }
        ]
    )
    second_response = json.dumps(
        [
            {
                "lemma": "alignment",
                "surface_form": "alignment",
                "meaning_en": "matching goals",
                "meaning_ru": "согласование",
                "usage_note": "Alignment matters.",
                "usage_note_translation": "Согласование важно.",
            },
            {
                "lemma": "trade-off",
                "surface_form": "trade-off",
                "meaning_en": "a balance between options",
                "meaning_ru": "компромисс",
                "usage_note": "There is a trade-off.",
                "usage_note_translation": "Есть компромисс.",
            },
            {
                "lemma": "baseline",
                "surface_form": "baseline",
                "meaning_en": "a reference result",
                "meaning_ru": "базовый уровень",
                "usage_note": "Compare it to a baseline.",
                "usage_note_translation": "Сравни это с базовым уровнем.",
            },
        ]
    )
    fake_llm = FakeLLM(first_response, second_response, second_response)
    text = "First paragraph about alignment.\n\nSecond paragraph about trade-offs and baselines."

    result = await ExtractionService(
        session_factory,
        fake_llm,
        max_words=2,
        chunk_max_chars=45,
        chunk_overlap_chars=0,
    ).extract_words(
        user_id,
        processing_job_id,
        text,
        profile_id=profile_id,
        profile_snapshot=profile_snapshot,
    )

    async with session_factory() as session:
        jobs = (
            await session.scalars(
                select(LLMExtractionJob).order_by(LLMExtractionJob.chunk_index.asc())
            )
        ).all()

    assert [item.lemma for item in result] == ["alignment", "trade-off"]
    assert len(jobs) == 3
    assert [job.chunk_index for job in jobs] == [1, 2, 3]
    assert all(job.chunk_count == 3 for job in jobs)


async def test_extract_rules_uses_matching_key_from_multi_section_payload(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    raw_response = json.dumps(
        {
            "words": [
                {
                    "lemma": "overfit",
                    "surface_form": "overfitting",
                    "meaning_en": "fit training data too closely",
                    "meaning_ru": "переобучаться",
                    "usage_note": "High depth is leading to overfitting.",
                    "usage_note_translation": "Большая глубина ведёт к переобучению.",
                }
            ],
            "rules": [
                {
                    "rule_en": "Use could be to give a possible explanation.",
                    "rule_ru": "Could be используется для возможного объяснения.",
                    "example": "One reason could be that I am not filtering features.",
                    "example_translation": (
                        "Одной причиной может быть то, "
                        "что я не фильтрую признаки."
                    ),
                }
            ],
        }
    )
    fake_llm = FakeLLM(raw_response)

    result = await ExtractionService(session_factory, fake_llm).extract_rules(
        user_id,
        processing_job_id,
        "One reason could be that I am not doing any filtering.",
    )

    assert [item.rule_en for item in result] == [
        "Use could be to give a possible explanation."
    ]


async def test_extract_phrases_accepts_nested_categorized_payload(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    raw_response = json.dumps(
        {
            "result": {
                "phrases": {
                    "question_frames": [
                        {
                            "phrase": "what type of",
                            "function": "asks for a category or kind",
                            "meaning_en": "asks which category is appropriate",
                            "meaning_ru": "какой тип",
                            "example": "What type of filtering should I do?",
                            "example_translation": "Какой тип фильтрации мне делать?",
                        }
                    ],
                    "hedging": [
                        {
                            "phrase": "could be that",
                            "function": "introduces a possible explanation",
                            "meaning_en": "shows uncertainty about a reason",
                            "meaning_ru": "может быть, что",
                            "example": "One reason could be that I am not filtering.",
                            "example_translation": (
                                "Одна причина может быть в том, "
                                "что я не делаю фильтрацию."
                            ),
                        }
                    ],
                }
            }
        }
    )
    fake_llm = FakeLLM(raw_response)

    result = await ExtractionService(session_factory, fake_llm).extract_phrases(
        user_id,
        processing_job_id,
        "What type of filtering should I do?",
    )

    assert [item.phrase for item in result] == ["what type of", "could be that"]


async def test_extract_phrases_skips_incomplete_items(
    session_factory: async_sessionmaker,
) -> None:
    user_id, processing_job_id = await _create_processing_job(session_factory)
    raw_response = json.dumps(
        [
            {
                "phrase": "stay on the cutting edge",
                "function": "talks about keeping up with the newest technology",
                "meaning_en": "to remain current with the most advanced developments",
                "example": "You have to give up a piece of your soul to stay on the cutting edge of tech.",
                "example_translation": "Приходится чем-то жертвовать, чтобы оставаться на переднем крае технологий.",
            },
            {
                "phrase": "on one hand",
                "function": "introduces one side of a contrast",
                "meaning_en": "used before the first of two contrasting points",
                "meaning_ru": "с одной стороны",
                "example": "On one hand, I am careful with my data.",
                "example_translation": "С одной стороны, я осторожен со своими данными.",
            },
        ]
    )
    fake_llm = FakeLLM(raw_response)

    result = await ExtractionService(session_factory, fake_llm).extract_phrases(
        user_id,
        processing_job_id,
        "On one hand, I am careful with data, but I want to stay on the cutting edge.",
    )

    async with session_factory() as session:
        job = await session.scalar(select(LLMExtractionJob))

    assert [item.phrase for item in result] == ["on one hand"]
    assert job is not None
    assert job.status == "done"
    assert job.error_message is None


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
        session.add(user)
        await session.flush()
        profile = await _create_profile(session, user.user_id)
        job = ProcessingJob(
            user=user,
            source_type=SourceType.REDDIT_POST.value,
            source_ref="https://www.reddit.com/r/test/comments/abc123/title/",
            profile_id=profile.profile_id,
            profile_snapshot=profile.profile_json,
        )
        session.add(job)
        await session.commit()
        await session.refresh(user)
        await session.refresh(job)
        return user.user_id, job.processing_job_id


async def _get_profile_snapshot(
    session_factory: async_sessionmaker,
    user_id: int,
) -> tuple[int, str]:
    async with session_factory() as session:
        profile = await ProfileService(session).require_active_profile(user_id)
        return profile.profile_id, profile.profile_json


async def _create_profile(session, user_id: int):
    return await ProfileService(session).upsert_profile(
        user_id,
        "B1. I want to read Reddit and machine learning discussions.",
        LearningProfilePayload(
            cefr_level="B1",
            level_confidence="high",
            goals_summary="Read Reddit and machine learning discussions.",
            focus_areas=["discussion phrases"],
            domain_interests=["Reddit", "machine learning"],
            preferred_item_types={"words": "high", "phrases": "high", "rules": "medium"},
            include=["domain vocabulary"],
            exclude=["very basic A1 words"],
            difficulty_policy="Mostly B1-B2 practical items.",
            extraction_guidance="Prioritize reusable discussion language.",
        ),
    )
