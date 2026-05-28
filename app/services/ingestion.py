from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    Phrase,
    PhraseExample,
    PhraseFunction,
    ProcessingJob,
    Rule,
    RuleExample,
    WordLemma,
    WordSurfaceForm,
    WordUsageNote,
)
from app.services.extraction import PhraseExtract, RuleExtract, WordExtract
from app.services.processing_jobs import MANUAL_POST_SOURCE_CODE, ProcessingJobService
from app.services.users import get_or_create_user


DEFAULT_COMMENTS_LIMIT = 20


class RedditTextFetcher(Protocol):
    async def fetch_post_text(self, url: str, comments_limit: int = DEFAULT_COMMENTS_LIMIT) -> str:
        ...


class StructuredExtractor(Protocol):
    async def extract_words(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[WordExtract]:
        ...

    async def extract_phrases(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[PhraseExtract]:
        ...

    async def extract_rules(
        self,
        user_id: int,
        processing_job_id: int,
        text: str,
    ) -> list[RuleExtract]:
        ...


class IngestionService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        reddit_service: RedditTextFetcher,
        extraction_service: StructuredExtractor,
        comments_limit: int = DEFAULT_COMMENTS_LIMIT,
    ) -> None:
        self._session_factory = session_factory
        self._reddit_service = reddit_service
        self._extraction_service = extraction_service
        self._comments_limit = comments_limit

    async def process_job(self, job_id: int) -> None:
        job = await self._get_job(job_id)
        if job.reddit_url == MANUAL_POST_SOURCE_CODE:
            source_text = (job.raw_text or "").strip()
            if not source_text:
                raise ValueError("Manual post text is empty")
        else:
            source_text = await self._reddit_service.fetch_post_text(
                job.reddit_url,
                comments_limit=self._comments_limit,
            )
            await self._save_raw_text(job_id, source_text)

        words = await self._extraction_service.extract_words(
            job.user_id,
            job.processing_job_id,
            source_text,
        )
        phrases = await self._extraction_service.extract_phrases(
            job.user_id,
            job.processing_job_id,
            source_text,
        )
        rules = await self._extraction_service.extract_rules(
            job.user_id,
            job.processing_job_id,
            source_text,
        )

        await self.add_words(job.user_id, words)
        await self.add_phrases(job.user_id, phrases)
        await self.add_rules(job.user_id, rules)

    async def add_words(self, user_id: int, words: list[WordExtract]) -> None:
        async with self._session_factory() as session:
            for word in words:
                lemma_text = _normalize(word.lemma).casefold()
                surface_form_text = _normalize(word.surface_form)
                meaning_en = _normalize(word.meaning_en)
                meaning_ru = _normalize(word.meaning_ru)
                usage_note_text = _normalize(word.usage_note)
                usage_note_translation = _normalize(word.usage_note_translation)
                if not _has_text(
                    lemma_text,
                    surface_form_text,
                    meaning_en,
                    meaning_ru,
                    usage_note_text,
                    usage_note_translation,
                ):
                    continue

                lemma = await session.scalar(
                    select(WordLemma)
                    .where(
                        WordLemma.user_id == user_id,
                        WordLemma.lemma == lemma_text,
                    )
                    .order_by(WordLemma.lemma_id.asc())
                    .limit(1)
                )
                if lemma is None:
                    lemma = WordLemma(user_id=user_id, lemma=lemma_text)
                    session.add(lemma)
                    await session.flush()

                surface_form = await session.scalar(
                    select(WordSurfaceForm)
                    .where(
                        WordSurfaceForm.lemma_id == lemma.lemma_id,
                        WordSurfaceForm.meaning_en == meaning_en,
                        WordSurfaceForm.meaning_ru == meaning_ru,
                    )
                    .order_by(WordSurfaceForm.surface_form_id.asc())
                    .limit(1)
                )
                if surface_form is None:
                    surface_form = WordSurfaceForm(
                        lemma_id=lemma.lemma_id,
                        surface_form=surface_form_text,
                        meaning_en=meaning_en,
                        meaning_ru=meaning_ru,
                    )
                    session.add(surface_form)
                    await session.flush()

                usage_note = await session.scalar(
                    select(WordUsageNote)
                    .where(
                        WordUsageNote.surface_form_id == surface_form.surface_form_id,
                        WordUsageNote.usage_note == usage_note_text,
                    )
                    .order_by(WordUsageNote.usage_note_id.asc())
                    .limit(1)
                )
                if usage_note is None:
                    session.add(
                        WordUsageNote(
                            surface_form_id=surface_form.surface_form_id,
                            usage_note=usage_note_text,
                            usage_note_translation=usage_note_translation,
                        )
                    )

            await session.commit()

    async def add_phrases(self, user_id: int, phrases: list[PhraseExtract]) -> None:
        async with self._session_factory() as session:
            for phrase_extract in phrases:
                phrase_text = _normalize(phrase_extract.phrase)
                function_text = _normalize(phrase_extract.function)
                meaning_en = _normalize(phrase_extract.meaning_en)
                meaning_ru = _normalize(phrase_extract.meaning_ru)
                example_text = _normalize(phrase_extract.example)
                example_translation = _normalize(phrase_extract.example_translation)
                if not _has_text(
                    phrase_text,
                    function_text,
                    meaning_en,
                    meaning_ru,
                    example_text,
                    example_translation,
                ):
                    continue

                phrase = await session.scalar(
                    select(Phrase)
                    .where(
                        Phrase.user_id == user_id,
                        Phrase.phrase == phrase_text,
                    )
                    .order_by(Phrase.phrase_id.asc())
                    .limit(1)
                )
                if phrase is None:
                    phrase = Phrase(user_id=user_id, phrase=phrase_text)
                    session.add(phrase)
                    await session.flush()

                phrase_function = await session.scalar(
                    select(PhraseFunction)
                    .where(
                        PhraseFunction.phrase_id == phrase.phrase_id,
                        PhraseFunction.function == function_text,
                        PhraseFunction.meaning_en == meaning_en,
                        PhraseFunction.meaning_ru == meaning_ru,
                    )
                    .order_by(PhraseFunction.function_id.asc())
                    .limit(1)
                )
                if phrase_function is None:
                    phrase_function = PhraseFunction(
                        phrase_id=phrase.phrase_id,
                        function=function_text,
                        meaning_en=meaning_en,
                        meaning_ru=meaning_ru,
                    )
                    session.add(phrase_function)
                    await session.flush()

                example = await session.scalar(
                    select(PhraseExample)
                    .where(
                        PhraseExample.function_id == phrase_function.function_id,
                        PhraseExample.example == example_text,
                    )
                    .order_by(PhraseExample.example_id.asc())
                    .limit(1)
                )
                if example is None:
                    session.add(
                        PhraseExample(
                            function_id=phrase_function.function_id,
                            example=example_text,
                            example_translation=example_translation,
                        )
                    )

            await session.commit()

    async def add_rules(self, user_id: int, rules: list[RuleExtract]) -> None:
        async with self._session_factory() as session:
            for rule_extract in rules:
                rule_en = _normalize(rule_extract.rule_en)
                rule_ru = _normalize(rule_extract.rule_ru)
                example_text = _normalize(rule_extract.example)
                example_translation = _normalize(rule_extract.example_translation)
                if not _has_text(rule_en, rule_ru, example_text, example_translation):
                    continue

                rule = await session.scalar(
                    select(Rule)
                    .where(
                        Rule.user_id == user_id,
                        Rule.rule_en == rule_en,
                        Rule.rule_ru == rule_ru,
                    )
                    .order_by(Rule.rule_id.asc())
                    .limit(1)
                )
                if rule is None:
                    rule = Rule(user_id=user_id, rule_en=rule_en, rule_ru=rule_ru)
                    session.add(rule)
                    await session.flush()

                example = await session.scalar(
                    select(RuleExample)
                    .where(
                        RuleExample.rule_id == rule.rule_id,
                        RuleExample.example == example_text,
                    )
                    .order_by(RuleExample.example_id.asc())
                    .limit(1)
                )
                if example is None:
                    session.add(
                        RuleExample(
                            rule_id=rule.rule_id,
                            example=example_text,
                            example_translation=example_translation,
                        )
                    )

            await session.commit()

    async def _get_job(self, job_id: int) -> ProcessingJob:
        async with self._session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise ValueError(f"Processing job {job_id} was not found")
            return job

    async def _save_raw_text(self, job_id: int, source_text: str) -> None:
        async with self._session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise ValueError(f"Processing job {job_id} was not found")
            job.raw_text = source_text
            await session.commit()


async def queue_reddit_url(
    session: AsyncSession,
    telegram_id: int,
    reddit_url: str,
) -> ProcessingJob:
    user = await get_or_create_user(session, telegram_id)
    if user is None:
        raise RuntimeError("Could not create user")

    result = await ProcessingJobService(session).queue_reddit_url(user.user_id, reddit_url)
    return result.job


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _has_text(*values: str) -> bool:
    return all(values)
