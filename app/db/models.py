from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ReviewScoreMixin:
    current_score: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default="1",
        nullable=False,
    )
    last_repetition: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class User(CreatedAtMixin, Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(
        BigInteger,
        unique=True,
        index=True,
        nullable=False,
    )
    username: Mapped[str | None] = mapped_column(String(255))
    first_name: Mapped[str | None] = mapped_column(String(255))
    last_activity: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    word_lemmas: Mapped[list[WordLemma]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    phrases: Mapped[list[Phrase]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    rules: Mapped[list[Rule]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    processing_jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    llm_extraction_jobs: Mapped[list[LLMExtractionJob]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    review_sessions: Mapped[list[ReviewSession]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class WordLemma(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "word_lemmas"
    __table_args__ = (Index("ix_word_lemmas_user_id_lemma", "user_id", "lemma"),)

    lemma_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lemma: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        index=True,
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="word_lemmas")
    surface_forms: Mapped[list[WordSurfaceForm]] = relationship(
        back_populates="lemma",
        cascade="all, delete-orphan",
    )


class WordSurfaceForm(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "word_surface_forms"

    surface_form_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    surface_form: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    meaning_en: Mapped[str] = mapped_column(Text, nullable=False)
    meaning_ru: Mapped[str] = mapped_column(Text, nullable=False)
    lemma_id: Mapped[int] = mapped_column(
        ForeignKey("word_lemmas.lemma_id"),
        index=True,
        nullable=False,
    )

    lemma: Mapped[WordLemma] = relationship(back_populates="surface_forms")
    usage_notes: Mapped[list[WordUsageNote]] = relationship(
        back_populates="surface_form",
        cascade="all, delete-orphan",
    )


class WordUsageNote(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "word_usage_notes"

    usage_note_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    usage_note: Mapped[str] = mapped_column(Text, nullable=False)
    usage_note_translation: Mapped[str] = mapped_column(Text, nullable=False)
    surface_form_id: Mapped[int] = mapped_column(
        ForeignKey("word_surface_forms.surface_form_id"),
        index=True,
        nullable=False,
    )

    surface_form: Mapped[WordSurfaceForm] = relationship(back_populates="usage_notes")


class Phrase(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "phrases"

    phrase_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phrase: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        index=True,
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="phrases")
    functions: Mapped[list[PhraseFunction]] = relationship(
        back_populates="phrase",
        cascade="all, delete-orphan",
    )


class PhraseFunction(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "phrase_functions"

    function_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    function: Mapped[str] = mapped_column(Text, nullable=False)
    meaning_en: Mapped[str] = mapped_column(Text, nullable=False)
    meaning_ru: Mapped[str] = mapped_column(Text, nullable=False)
    phrase_id: Mapped[int] = mapped_column(
        ForeignKey("phrases.phrase_id"),
        index=True,
        nullable=False,
    )

    phrase: Mapped[Phrase] = relationship(back_populates="functions")
    examples: Mapped[list[PhraseExample]] = relationship(
        back_populates="function",
        cascade="all, delete-orphan",
    )


class PhraseExample(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "phrase_examples"

    example_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    example: Mapped[str] = mapped_column(Text, nullable=False)
    example_translation: Mapped[str] = mapped_column(Text, nullable=False)
    function_id: Mapped[int] = mapped_column(
        ForeignKey("phrase_functions.function_id"),
        index=True,
        nullable=False,
    )

    function: Mapped[PhraseFunction] = relationship(back_populates="examples")


class Rule(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "rules"

    rule_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_en: Mapped[str] = mapped_column(Text, nullable=False)
    rule_ru: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        index=True,
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="rules")
    examples: Mapped[list[RuleExample]] = relationship(
        back_populates="rule",
        cascade="all, delete-orphan",
    )


class RuleExample(CreatedAtMixin, ReviewScoreMixin, Base):
    __tablename__ = "rule_examples"

    example_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    example: Mapped[str] = mapped_column(Text, nullable=False)
    example_translation: Mapped[str] = mapped_column(Text, nullable=False)
    rule_id: Mapped[int] = mapped_column(
        ForeignKey("rules.rule_id"),
        index=True,
        nullable=False,
    )

    rule: Mapped[Rule] = relationship(back_populates="examples")


class LLMExtractionJob(CreatedAtMixin, Base):
    __tablename__ = "llm_extraction_jobs"

    llm_job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        index=True,
        nullable=False,
    )
    processing_job_id: Mapped[int | None] = mapped_column(
        ForeignKey("processing_jobs.processing_job_id"),
        index=True,
    )
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[str | None] = mapped_column(Text)
    parsed_response: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32),
        default="created",
        server_default="created",
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="llm_extraction_jobs")
    processing_job: Mapped[ProcessingJob | None] = relationship(
        back_populates="llm_extraction_jobs",
    )


class ProcessingJob(CreatedAtMixin, Base):
    __tablename__ = "processing_jobs"

    processing_job_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        index=True,
        nullable=False,
    )
    reddit_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default="queued",
        server_default="queued",
        nullable=False,
    )
    raw_text: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="processing_jobs")
    llm_extraction_jobs: Mapped[list[LLMExtractionJob]] = relationship(
        back_populates="processing_job",
    )


class ReviewSession(CreatedAtMixin, Base):
    __tablename__ = "review_sessions"

    review_session_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        index=True,
        nullable=False,
    )
    session_type: Mapped[str] = mapped_column(String(32), nullable=False)
    items: Mapped[str] = mapped_column(Text, nullable=False)
    current_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        server_default="active",
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="review_sessions")
