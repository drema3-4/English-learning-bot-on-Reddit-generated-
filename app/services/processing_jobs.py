from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ProcessingJob, UserLearningProfile
from app.services.profiles import MissingLearningProfileError, ProfileService
from app.services.sources.types import SourceType
from app.utils.reddit_url import extract_reddit_post_ref


ACTIVE_JOB_STATUSES = {"queued", "processing"}
MAX_MANUAL_POST_TEXT_CHARS = 25_000


class ManualPostTextError(ValueError):
    pass


@dataclass(frozen=True)
class QueueProcessingJobResult:
    job: ProcessingJob
    created: bool


class ProcessingJobService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def queue_source(
        self,
        user_id: int,
        source_type: SourceType,
        source_ref: str | None,
        raw_text: str | None = None,
        source_metadata: str | None = None,
        profile: UserLearningProfile | None = None,
        require_profile: bool = True,
    ) -> QueueProcessingJobResult:
        active_job = await self.get_active_job(user_id)
        if active_job is not None:
            return QueueProcessingJobResult(job=active_job, created=False)

        if profile is None and require_profile:
            profile = await ProfileService(self._session).get_active_profile(user_id)
            if profile is None:
                raise MissingLearningProfileError("Active learning profile is missing")

        normalized_source_ref = source_ref
        normalized_raw_text = raw_text
        if source_type == SourceType.MANUAL_TEXT:
            normalized_source_ref = None
            normalized_raw_text = _normalize_manual_text(raw_text or "")
            if not normalized_raw_text:
                raise ManualPostTextError("Manual text is empty")
        elif source_type == SourceType.REDDIT_POST:
            if not source_ref:
                raise ValueError("Reddit source_ref is empty")
            ref = extract_reddit_post_ref(source_ref)
            normalized_source_ref = ref.normalized_url
            normalized_raw_text = None
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        job = ProcessingJob(
            user_id=user_id,
            source_type=source_type.value,
            source_ref=normalized_source_ref,
            raw_text=normalized_raw_text,
            source_metadata=source_metadata,
            profile_id=profile.profile_id if profile is not None else None,
            profile_snapshot=profile.profile_json if profile is not None else None,
            status="queued",
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return QueueProcessingJobResult(job=job, created=True)

    async def queue_reddit_post(
        self,
        user_id: int,
        reddit_url: str,
        source_metadata: str | None = None,
        profile: UserLearningProfile | None = None,
        require_profile: bool = True,
    ) -> QueueProcessingJobResult:
        return await self.queue_source(
            user_id=user_id,
            source_type=SourceType.REDDIT_POST,
            source_ref=reddit_url,
            source_metadata=source_metadata,
            profile=profile,
            require_profile=require_profile,
        )

    async def queue_manual_text(
        self,
        user_id: int,
        text: str,
        profile: UserLearningProfile | None = None,
        require_profile: bool = True,
    ) -> QueueProcessingJobResult:
        return await self.queue_source(
            user_id=user_id,
            source_type=SourceType.MANUAL_TEXT,
            source_ref=None,
            raw_text=text,
            profile=profile,
            require_profile=require_profile,
        )

    async def get_active_job(self, user_id: int) -> ProcessingJob | None:
        return await self._session.scalar(
            select(ProcessingJob)
            .where(
                ProcessingJob.user_id == user_id,
                ProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
            )
            .order_by(ProcessingJob.created_at.asc(), ProcessingJob.processing_job_id.asc())
            .limit(1)
        )

    async def get_latest_job(self, user_id: int) -> ProcessingJob | None:
        return await self._session.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.user_id == user_id)
            .order_by(ProcessingJob.created_at.desc(), ProcessingJob.processing_job_id.desc())
            .limit(1)
        )


def _normalize_manual_text(text: str) -> str:
    normalized = text.strip()
    if len(normalized) > MAX_MANUAL_POST_TEXT_CHARS:
        normalized = normalized[:MAX_MANUAL_POST_TEXT_CHARS].rstrip()
    return normalized
