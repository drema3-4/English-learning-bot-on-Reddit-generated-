from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LLMProfileJob, UserLearningProfile
from app.services.profile_schemas import LearningProfilePayload
from app.services.profiles import ProfileGenerationError, ProfileService
from app.utils.json_parse import parse_json_object


PROFILE_GENERATION_PROMPT_TEMPLATE = """You create a learning profile for an English-learning Telegram bot.

The user may write in Russian or English. Extract their English level, learning goals, topics, and priorities.

Return only one valid JSON object. Do not add markdown. Do not add explanations outside JSON.

Allowed cefr_level values:
A1, A2, B1, B2, C1, C2, unknown

Rules:
- If the user gives a CEFR level, preserve it.
- If the level is unclear, use "unknown".
- Do not invent very specific goals that the user did not imply.
- Prefer practical, reusable extraction guidance.
- The profile will be used to choose words, phrases and grammar rules from source texts.
- Keep extraction_guidance concise and safe.

JSON shape:
{
  "cefr_level": "B1",
  "level_confidence": "low | medium | high",
  "goals_summary": "...",
  "focus_areas": ["..."],
  "domain_interests": ["..."],
  "preferred_item_types": {
    "words": "low | medium | high",
    "phrases": "low | medium | high",
    "rules": "low | medium | high"
  },
  "include": ["..."],
  "exclude": ["..."],
  "difficulty_policy": "...",
  "extraction_guidance": "..."
}

User input:
{raw_user_input}
"""


class JSONCompleter(Protocol):
    async def complete_json(self, prompt: str) -> str:
        ...


class ProfileGenerationService:
    def __init__(
        self,
        session: AsyncSession,
        llm_client: JSONCompleter,
        max_input_chars: int = 3000,
    ) -> None:
        self._session = session
        self._llm_client = llm_client
        self._max_input_chars = max_input_chars

    async def generate_profile(
        self,
        user_id: int,
        raw_user_input: str,
    ) -> UserLearningProfile:
        normalized_input = _normalize_input(raw_user_input)
        if not normalized_input:
            raise ProfileGenerationError("Profile input is empty")
        normalized_input = normalized_input[: self._max_input_chars].rstrip()
        prompt = build_profile_generation_prompt(normalized_input)
        llm_job = await self._create_job(user_id, normalized_input, prompt)
        raw_response: str | None = None

        try:
            raw_response = await self._llm_client.complete_json(prompt)
            parsed = parse_json_object(raw_response)
            payload = LearningProfilePayload.model_validate(parsed)
            profile = await ProfileService(self._session).upsert_profile(
                user_id,
                normalized_input,
                payload,
            )
        except Exception as exc:  # noqa: BLE001
            await self._mark_job_failed(llm_job.llm_profile_job_id, raw_response, str(exc))
            raise ProfileGenerationError(str(exc)) from exc

        await self._mark_job_done(llm_job.llm_profile_job_id, raw_response, payload)
        return profile

    async def _create_job(
        self,
        user_id: int,
        input_text: str,
        prompt_text: str,
    ) -> LLMProfileJob:
        llm_job = LLMProfileJob(
            user_id=user_id,
            input_text=input_text,
            prompt_text=prompt_text,
            status="processing",
            started_at=datetime.now(UTC),
        )
        self._session.add(llm_job)
        await self._session.commit()
        await self._session.refresh(llm_job)
        return llm_job

    async def _mark_job_done(
        self,
        llm_profile_job_id: int,
        raw_response: str,
        payload: LearningProfilePayload,
    ) -> None:
        llm_job = await self._session.get(LLMProfileJob, llm_profile_job_id)
        if llm_job is None:
            return
        llm_job.status = "done"
        llm_job.raw_response = raw_response
        llm_job.parsed_response = json.dumps(payload.model_dump(), ensure_ascii=False)
        llm_job.error_message = None
        llm_job.finished_at = datetime.now(UTC)
        await self._session.commit()

    async def _mark_job_failed(
        self,
        llm_profile_job_id: int,
        raw_response: str | None,
        error_message: str,
    ) -> None:
        llm_job = await self._session.get(LLMProfileJob, llm_profile_job_id)
        if llm_job is None:
            return
        llm_job.status = "failed"
        llm_job.raw_response = raw_response
        llm_job.error_message = error_message[:4000]
        llm_job.finished_at = datetime.now(UTC)
        await self._session.commit()


def build_profile_generation_prompt(raw_user_input: str) -> str:
    return PROFILE_GENERATION_PROMPT_TEMPLATE.replace("{raw_user_input}", raw_user_input)


def _normalize_input(raw_user_input: str) -> str:
    return " ".join((raw_user_input or "").split())
