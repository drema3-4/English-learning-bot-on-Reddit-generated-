"""user learning profiles

Revision ID: 20260531_0002
Revises: 20260527_0001
Create Date: 2026-05-31
"""

from alembic import op
import sqlalchemy as sa


revision = "20260531_0002"
down_revision = "20260527_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_learning_profiles",
        sa.Column("profile_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("raw_user_input", sa.Text(), nullable=False),
        sa.Column("cefr_level", sa.String(length=16), nullable=False),
        sa.Column("goals_summary", sa.Text(), nullable=False),
        sa.Column("profile_json", sa.Text(), nullable=False),
        sa.Column("prompt_profile", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("profile_id"),
    )
    op.create_index(
        "ix_user_learning_profiles_user_id",
        "user_learning_profiles",
        ["user_id"],
        unique=True,
    )

    op.create_table(
        "user_bot_states",
        sa.Column("state_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("state_id"),
    )
    op.create_index(
        "ix_user_bot_states_user_id",
        "user_bot_states",
        ["user_id"],
        unique=True,
    )

    op.create_table(
        "llm_profile_jobs",
        sa.Column("llm_profile_job_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("parsed_response", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="created", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("llm_profile_job_id"),
    )
    op.create_index(
        op.f("ix_llm_profile_jobs_user_id"),
        "llm_profile_jobs",
        ["user_id"],
        unique=False,
    )

    with op.batch_alter_table("processing_jobs") as batch_op:
        batch_op.add_column(sa.Column("profile_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("profile_snapshot", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_processing_jobs_profile_id_user_learning_profiles",
            "user_learning_profiles",
            ["profile_id"],
            ["profile_id"],
        )
        batch_op.create_index(
            op.f("ix_processing_jobs_profile_id"),
            ["profile_id"],
            unique=False,
        )

    with op.batch_alter_table("llm_extraction_jobs") as batch_op:
        batch_op.add_column(sa.Column("profile_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("profile_snapshot", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("chunk_index", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("chunk_count", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_llm_extraction_jobs_profile_id_user_learning_profiles",
            "user_learning_profiles",
            ["profile_id"],
            ["profile_id"],
        )
        batch_op.create_index(
            op.f("ix_llm_extraction_jobs_profile_id"),
            ["profile_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("llm_extraction_jobs") as batch_op:
        batch_op.drop_index(op.f("ix_llm_extraction_jobs_profile_id"))
        batch_op.drop_constraint(
            "fk_llm_extraction_jobs_profile_id_user_learning_profiles",
            type_="foreignkey",
        )
        batch_op.drop_column("chunk_count")
        batch_op.drop_column("chunk_index")
        batch_op.drop_column("profile_snapshot")
        batch_op.drop_column("profile_id")

    with op.batch_alter_table("processing_jobs") as batch_op:
        batch_op.drop_index(op.f("ix_processing_jobs_profile_id"))
        batch_op.drop_constraint(
            "fk_processing_jobs_profile_id_user_learning_profiles",
            type_="foreignkey",
        )
        batch_op.drop_column("profile_snapshot")
        batch_op.drop_column("profile_id")

    op.drop_index(op.f("ix_llm_profile_jobs_user_id"), table_name="llm_profile_jobs")
    op.drop_table("llm_profile_jobs")
    op.drop_index("ix_user_bot_states_user_id", table_name="user_bot_states")
    op.drop_table("user_bot_states")
    op.drop_index("ix_user_learning_profiles_user_id", table_name="user_learning_profiles")
    op.drop_table("user_learning_profiles")
