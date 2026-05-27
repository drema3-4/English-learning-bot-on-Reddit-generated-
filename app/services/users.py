from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_telegram_id(self, telegram_id: int) -> User | None:
        return await self._session.scalar(select(User).where(User.telegram_id == telegram_id))

    async def get_or_create(
        self,
        telegram_id: int,
        max_users: int | None = None,
        username: str | None = None,
        first_name: str | None = None,
    ) -> User | None:
        user = await self.get_by_telegram_id(telegram_id)
        now = datetime.now(UTC)
        if user is not None:
            user.username = username or user.username
            user.first_name = first_name or user.first_name
            user.last_activity = now
            await self._session.commit()
            await self._session.refresh(user)
            return user

        if max_users is not None:
            user_count = await self._session.scalar(select(func.count()).select_from(User))
            if user_count is not None and user_count >= max_users:
                return None

        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_activity=now,
        )
        self._session.add(user)
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def ensure_allowed(
        self,
        telegram_id: int,
        max_users: int,
        username: str | None = None,
        first_name: str | None = None,
    ) -> User | None:
        return await self.get_or_create(
            telegram_id,
            max_users=max_users,
            username=username,
            first_name=first_name,
        )


async def get_or_create_user(
    session: AsyncSession,
    telegram_id: int,
    max_users: int | None = None,
    username: str | None = None,
    first_name: str | None = None,
) -> User | None:
    return await UserService(session).get_or_create(
        telegram_id,
        max_users=max_users,
        username=username,
        first_name=first_name,
    )


async def get_user_by_telegram_id(session: AsyncSession, telegram_id: int) -> User | None:
    return await UserService(session).get_by_telegram_id(telegram_id)


async def ensure_user_allowed(
    session: AsyncSession,
    telegram_id: int,
    max_users: int,
    username: str | None = None,
    first_name: str | None = None,
) -> bool:
    user = await UserService(session).ensure_allowed(
        telegram_id,
        max_users=max_users,
        username=username,
        first_name=first_name,
    )
    return user is not None
