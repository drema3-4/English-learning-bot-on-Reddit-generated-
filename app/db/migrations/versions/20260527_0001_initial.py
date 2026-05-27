"""initial schema

Revision ID: 20260527_0001
Revises:
Create Date: 2026-05-27
"""

from alembic import op
import sqlalchemy as sa


revision = "20260527_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("first_name", sa.String(length=255), nullable=True),
        sa.Column("last_activity", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index(op.f("ix_users_telegram_id"), "users", ["telegram_id"], unique=True)

    op.create_table(
        "phrases",
        sa.Column("phrase_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("phrase", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("phrase_id"),
    )
    op.create_index(op.f("ix_phrases_phrase"), "phrases", ["phrase"], unique=False)
    op.create_index(op.f("ix_phrases_user_id"), "phrases", ["user_id"], unique=False)

    op.create_table(
        "processing_jobs",
        sa.Column("processing_job_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("reddit_url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="queued", nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("processing_job_id"),
    )
    op.create_index(op.f("ix_processing_jobs_user_id"), "processing_jobs", ["user_id"], unique=False)

    op.create_table(
        "review_sessions",
        sa.Column("review_session_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_type", sa.String(length=32), nullable=False),
        sa.Column("items", sa.Text(), nullable=False),
        sa.Column("current_index", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("review_session_id"),
    )
    op.create_index(op.f("ix_review_sessions_user_id"), "review_sessions", ["user_id"], unique=False)

    op.create_table(
        "rules",
        sa.Column("rule_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("rule_en", sa.Text(), nullable=False),
        sa.Column("rule_ru", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("rule_id"),
    )
    op.create_index(op.f("ix_rules_user_id"), "rules", ["user_id"], unique=False)

    op.create_table(
        "word_lemmas",
        sa.Column("lemma_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("lemma", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("lemma_id"),
    )
    op.create_index(op.f("ix_word_lemmas_lemma"), "word_lemmas", ["lemma"], unique=False)
    op.create_index(op.f("ix_word_lemmas_user_id"), "word_lemmas", ["user_id"], unique=False)
    op.create_index("ix_word_lemmas_user_id_lemma", "word_lemmas", ["user_id", "lemma"], unique=False)

    op.create_table(
        "llm_extraction_jobs",
        sa.Column("llm_job_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("processing_job_id", sa.Integer(), nullable=True),
        sa.Column("job_type", sa.String(length=32), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("parsed_response", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="created", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["processing_job_id"], ["processing_jobs.processing_job_id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("llm_job_id"),
    )
    op.create_index(
        op.f("ix_llm_extraction_jobs_processing_job_id"),
        "llm_extraction_jobs",
        ["processing_job_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_llm_extraction_jobs_user_id"),
        "llm_extraction_jobs",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "phrase_functions",
        sa.Column("function_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("function", sa.Text(), nullable=False),
        sa.Column("meaning_en", sa.Text(), nullable=False),
        sa.Column("meaning_ru", sa.Text(), nullable=False),
        sa.Column("phrase_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["phrase_id"], ["phrases.phrase_id"]),
        sa.PrimaryKeyConstraint("function_id"),
    )
    op.create_index(op.f("ix_phrase_functions_phrase_id"), "phrase_functions", ["phrase_id"], unique=False)

    op.create_table(
        "rule_examples",
        sa.Column("example_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("example", sa.Text(), nullable=False),
        sa.Column("example_translation", sa.Text(), nullable=False),
        sa.Column("rule_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["rule_id"], ["rules.rule_id"]),
        sa.PrimaryKeyConstraint("example_id"),
    )
    op.create_index(op.f("ix_rule_examples_rule_id"), "rule_examples", ["rule_id"], unique=False)

    op.create_table(
        "word_surface_forms",
        sa.Column("surface_form_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("surface_form", sa.String(length=255), nullable=False),
        sa.Column("meaning_en", sa.Text(), nullable=False),
        sa.Column("meaning_ru", sa.Text(), nullable=False),
        sa.Column("lemma_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["lemma_id"], ["word_lemmas.lemma_id"]),
        sa.PrimaryKeyConstraint("surface_form_id"),
    )
    op.create_index(
        op.f("ix_word_surface_forms_lemma_id"),
        "word_surface_forms",
        ["lemma_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_word_surface_forms_surface_form"),
        "word_surface_forms",
        ["surface_form"],
        unique=False,
    )

    op.create_table(
        "phrase_examples",
        sa.Column("example_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("example", sa.Text(), nullable=False),
        sa.Column("example_translation", sa.Text(), nullable=False),
        sa.Column("function_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["function_id"], ["phrase_functions.function_id"]),
        sa.PrimaryKeyConstraint("example_id"),
    )
    op.create_index(op.f("ix_phrase_examples_function_id"), "phrase_examples", ["function_id"], unique=False)

    op.create_table(
        "word_usage_notes",
        sa.Column("usage_note_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("usage_note", sa.Text(), nullable=False),
        sa.Column("usage_note_translation", sa.Text(), nullable=False),
        sa.Column("surface_form_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_score", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_repetition", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["surface_form_id"], ["word_surface_forms.surface_form_id"]),
        sa.PrimaryKeyConstraint("usage_note_id"),
    )
    op.create_index(
        op.f("ix_word_usage_notes_surface_form_id"),
        "word_usage_notes",
        ["surface_form_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_word_usage_notes_surface_form_id"), table_name="word_usage_notes")
    op.drop_table("word_usage_notes")
    op.drop_index(op.f("ix_phrase_examples_function_id"), table_name="phrase_examples")
    op.drop_table("phrase_examples")
    op.drop_index(op.f("ix_word_surface_forms_surface_form"), table_name="word_surface_forms")
    op.drop_index(op.f("ix_word_surface_forms_lemma_id"), table_name="word_surface_forms")
    op.drop_table("word_surface_forms")
    op.drop_index(op.f("ix_rule_examples_rule_id"), table_name="rule_examples")
    op.drop_table("rule_examples")
    op.drop_index(op.f("ix_phrase_functions_phrase_id"), table_name="phrase_functions")
    op.drop_table("phrase_functions")
    op.drop_index(op.f("ix_llm_extraction_jobs_user_id"), table_name="llm_extraction_jobs")
    op.drop_index(op.f("ix_llm_extraction_jobs_processing_job_id"), table_name="llm_extraction_jobs")
    op.drop_table("llm_extraction_jobs")
    op.drop_index("ix_word_lemmas_user_id_lemma", table_name="word_lemmas")
    op.drop_index(op.f("ix_word_lemmas_user_id"), table_name="word_lemmas")
    op.drop_index(op.f("ix_word_lemmas_lemma"), table_name="word_lemmas")
    op.drop_table("word_lemmas")
    op.drop_index(op.f("ix_rules_user_id"), table_name="rules")
    op.drop_table("rules")
    op.drop_index(op.f("ix_review_sessions_user_id"), table_name="review_sessions")
    op.drop_table("review_sessions")
    op.drop_index(op.f("ix_processing_jobs_user_id"), table_name="processing_jobs")
    op.drop_table("processing_jobs")
    op.drop_index(op.f("ix_phrases_user_id"), table_name="phrases")
    op.drop_index(op.f("ix_phrases_phrase"), table_name="phrases")
    op.drop_table("phrases")
    op.drop_index(op.f("ix_users_telegram_id"), table_name="users")
    op.drop_table("users")
