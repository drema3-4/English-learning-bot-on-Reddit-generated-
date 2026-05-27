FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock alembic.ini docker-entrypoint.sh ./
COPY app ./app

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

CMD ["sh", "./docker-entrypoint.sh"]
