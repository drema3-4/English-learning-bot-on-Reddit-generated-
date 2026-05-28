from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ProcessingJob
from app.utils.reddit_url import extract_reddit_post_ref


ACTIVE_JOB_STATUSES = {"queued", "processing"}
MANUAL_POST_SOURCE_CODE = "about:blank"
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

    async def queue_reddit_url(
        self,
        user_id: int,
        reddit_url: str,
    ) -> QueueProcessingJobResult:
        active_job = await self.get_active_job(user_id)
        if active_job is not None:
            return QueueProcessingJobResult(job=active_job, created=False)

        ref = extract_reddit_post_ref(reddit_url)
        job = ProcessingJob(
            user_id=user_id,
            reddit_url=ref.normalized_url,
            status="queued",
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return QueueProcessingJobResult(job=job, created=True)

    async def queue_manual_text(
        self,
        user_id: int,
        post_text: str,
    ) -> QueueProcessingJobResult:
        active_job = await self.get_active_job(user_id)
        if active_job is not None:
            return QueueProcessingJobResult(job=active_job, created=False)

        normalized_text = _normalize_manual_post_text(post_text)
        if not normalized_text:
            raise ManualPostTextError("Manual post text is empty")

        job = ProcessingJob(
            user_id=user_id,
            reddit_url=MANUAL_POST_SOURCE_CODE,
            raw_text=normalized_text,
            status="queued",
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return QueueProcessingJobResult(job=job, created=True)

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


def _normalize_manual_post_text(text: str) -> str:
    normalized = text.strip()
    if len(normalized) > MAX_MANUAL_POST_TEXT_CHARS:
        normalized = normalized[:MAX_MANUAL_POST_TEXT_CHARS].rstrip()
    return normalized
