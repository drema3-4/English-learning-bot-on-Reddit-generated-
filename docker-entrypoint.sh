#!/bin/sh
set -e

poetry run alembic upgrade head
exec poetry run bot
