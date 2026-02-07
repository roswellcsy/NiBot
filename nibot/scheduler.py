"""Cron scheduler -- periodic task trigger via MessageBus."""

from __future__ import annotations

import asyncio
from datetime import datetime

from croniter import croniter

from nibot.bus import MessageBus
from nibot.config import ScheduledJob
from nibot.log import logger
from nibot.types import Envelope


class SchedulerManager:
    """Fire scheduled jobs by publishing to the inbound bus at cron-specified times.

    The scheduler is a pure message producer -- it knows nothing about AgentLoop.
    Runtime source of truth: self._jobs dict.
    Persistent source of truth: config.json schedules field (written by ScheduleTool).
    """

    def __init__(self, bus: MessageBus, jobs: list[ScheduledJob]) -> None:
        self._bus = bus
        self._jobs: dict[str, ScheduledJob] = {j.id: j for j in jobs if j.id}
        self._last_check = datetime.now()
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(60)
            now = datetime.now()
            for job in list(self._jobs.values()):
                if not job.enabled:
                    continue
                try:
                    cron = croniter(job.cron, self._last_check)
                    next_run = cron.get_next(datetime)
                    if next_run <= now:
                        await self._fire(job)
                except Exception as e:
                    logger.error(f"Scheduler job '{job.id}' error: {e}")
            self._last_check = now

    async def _fire(self, job: ScheduledJob) -> None:
        logger.info(f"Scheduler firing job: {job.id}")
        await self._bus.publish_inbound(
            Envelope(
                channel=job.channel,
                chat_id=job.chat_id,
                sender_id="scheduler",
                content=job.prompt,
                metadata={"scheduled": True, "job_id": job.id},
            )
        )

    def add(self, job: ScheduledJob) -> None:
        self._jobs[job.id] = job

    def remove(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    def list_jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    def stop(self) -> None:
        self._running = False
