from __future__ import annotations

from app.bot import main as bot_main
from app.config import Settings


def test_main_runs_migrations_before_starting_async_bot(monkeypatch) -> None:
    calls: list[str] = []
    settings = Settings(telegram_bot_token="test-token")

    async def fake_async_main(received_settings: Settings | None = None) -> None:
        assert calls == ["migrations"]
        assert received_settings == settings
        calls.append("async_main")

    monkeypatch.setattr(bot_main, "get_settings", lambda: settings)
    monkeypatch.setattr(bot_main, "run_migrations", lambda: calls.append("migrations"))
    monkeypatch.setattr(bot_main, "async_main", fake_async_main)

    bot_main.main()

    assert calls == ["migrations", "async_main"]
