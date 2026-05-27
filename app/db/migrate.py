from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.config import get_settings
from app.db.session import ensure_sqlite_dir


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_migrations() -> None:
    settings = get_settings()
    ensure_sqlite_dir(settings.database_url)
    config = Config(str(project_root() / "alembic.ini"))
    config.set_main_option("script_location", str(project_root() / "app" / "db" / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(config, "head")


def main() -> None:
    run_migrations()


if __name__ == "__main__":
    main()

