from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UserBotState, UserLearningProfile
from app.services.profile_prompts import render_profile_for_prompt
from app.services.profile_schemas import LearningProfilePayload


AWAITING_PROFILE_INPUT = "awaiting_profile_input"


class MissingLearningProfileError(ValueError):
    pass


class ProfileGenerationError(ValueError):
    pass


class ProfileService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_active_profile(self, user_id: int) -> UserLearningProfile | None:
        return await self._session.scalar(
            select(UserLearningProfile)
            .where(
                UserLearningProfile.user_id == user_id,
                UserLearningProfile.status == "active",
            )
            .order_by(UserLearningProfile.profile_id.asc())
            .limit(1)
        )

    async def require_active_profile(self, user_id: int) -> UserLearningProfile:
        profile = await self.get_active_profile(user_id)
        if profile is None:
            raise MissingLearningProfileError("Active learning profile is missing")
        return profile

    async def upsert_profile(
        self,
        user_id: int,
        raw_user_input: str,
        payload: LearningProfilePayload,
    ) -> UserLearningProfile:
        profile = await self._session.scalar(
            select(UserLearningProfile)
            .where(UserLearningProfile.user_id == user_id)
            .order_by(UserLearningProfile.profile_id.asc())
            .limit(1)
        )
        profile_json = json.dumps(payload.model_dump(), ensure_ascii=False)
        prompt_profile = render_profile_for_prompt(payload)
        raw_user_input = " ".join(raw_user_input.split())

        if profile is None:
            profile = UserLearningProfile(
                user_id=user_id,
                raw_user_input=raw_user_input,
                cefr_level=payload.cefr_level,
                goals_summary=payload.goals_summary,
                profile_json=profile_json,
                prompt_profile=prompt_profile,
                status="active",
            )
            self._session.add(profile)
        else:
            profile.raw_user_input = raw_user_input
            profile.cefr_level = payload.cefr_level
            profile.goals_summary = payload.goals_summary
            profile.profile_json = profile_json
            profile.prompt_profile = prompt_profile
            profile.status = "active"
            profile.error_message = None
            profile.updated_at = datetime.now(UTC)

        await self._session.commit()
        await self._session.refresh(profile)
        return profile

    async def get_state(self, user_id: int) -> UserBotState | None:
        return await self._session.scalar(
            select(UserBotState)
            .where(UserBotState.user_id == user_id)
            .order_by(UserBotState.state_id.asc())
            .limit(1)
        )

    async def set_awaiting_profile_input(self, user_id: int, reason: str) -> None:
        state = await self.get_state(user_id)
        payload = json.dumps({"reason": reason}, ensure_ascii=False)
        if state is None:
            self._session.add(
                UserBotState(
                    user_id=user_id,
                    state=AWAITING_PROFILE_INPUT,
                    payload=payload,
                )
            )
        else:
            state.state = AWAITING_PROFILE_INPUT
            state.payload = payload
            state.updated_at = datetime.now(UTC)
        await self._session.commit()

    async def clear_state(self, user_id: int) -> None:
        await self._session.execute(delete(UserBotState).where(UserBotState.user_id == user_id))
        await self._session.commit()
