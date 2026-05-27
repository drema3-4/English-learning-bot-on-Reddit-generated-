from __future__ import annotations

import asyncio
import contextlib

from aiogram import Bot, Dispatcher

from app.bot.handlers import router
from app.config import get_settings
from app.db.migrate import run_migrations
from app.db.session import create_session_factory
from app.workers.processing_loop import processing_loop


async def async_main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    run_migrations()
    session_factory = create_session_factory(settings)

    bot = Bot(settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    worker_task = asyncio.create_task(processing_loop(settings, session_factory, bot))
    try:
        await dispatcher.start_polling(
            bot,
            settings=settings,
            session_factory=session_factory,
        )
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        await bot.session.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

