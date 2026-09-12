"""Shared asyncio monitoring scheduler (PyQt-free).

Orchestrates per-tab monitoring cycles off the UI thread:
- one asyncio loop in a daemon thread
- sync ML/pandas work via a shared ThreadPoolExecutor
- deterministic overlap policy (skip if previous cycle still running)
- structured CycleMetrics for portfolio / ops latency claims
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("STDMS.AsyncPipeline")

ComputeFn = Callable[[], Any]
ResultFn = Callable[[str, Any, "CycleMetrics"], None]
DueFn = Callable[[], bool]


@dataclass
class CycleMetrics:
    tab_id: str
    started_at: float
    queue_wait_ms: float = 0.0
    compute_ms: float = 0.0
    total_ms: float = 0.0
    status: str = "ok"
    error: str = ""
    schedule_type: str = "Continuous"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _TabRegistration:
    tab_id: str
    interval_ms: int
    compute_fn: ComputeFn
    on_result: Optional[ResultFn]
    schedule_type: str = "Continuous"
    due_fn: Optional[DueFn] = None
    task: Optional[asyncio.Task] = None
    busy: bool = False
    stop: bool = False


class AsyncMonitoringPipeline:
    """Process-wide scheduler for custom-tab monitoring cycles."""

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        metrics_maxlen: int = 500,
    ):
        cpu = os.cpu_count() or 2
        self._max_workers = int(max_workers) if max_workers else min(8, cpu + 2)
        self._metrics: deque[CycleMetrics] = deque(maxlen=max(50, int(metrics_maxlen)))
        self._tabs: dict[str, _TabRegistration] = {}
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._started = threading.Event()
        self._stop = threading.Event()

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive() and self._started.is_set()

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop.clear()
            self._started.clear()
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="stdms-mon",
            )
            self._thread = threading.Thread(
                target=self._run_loop,
                name="stdms-async-pipeline",
                daemon=True,
            )
            self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("AsyncMonitoringPipeline failed to start event loop")
        logger.info(
            "AsyncMonitoringPipeline started (max_workers=%d)",
            self._max_workers,
        )

    def stop(self, *, wait: bool = True) -> None:
        with self._lock:
            regs = list(self._tabs.values())
            for reg in regs:
                reg.stop = True
            loop = self._loop
            self._stop.set()
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
            # Cancel tab tasks
            def _cancel_all():
                for reg in list(self._tabs.values()):
                    if reg.task and not reg.task.done():
                        reg.task.cancel()

            loop.call_soon_threadsafe(_cancel_all)
            loop.call_soon_threadsafe(loop.stop)
        if wait and self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None
        with self._lock:
            self._tabs.clear()
            self._loop = None
            self._thread = None
            self._started.clear()
        logger.info("AsyncMonitoringPipeline stopped")

    def register_tab(
        self,
        tab_id: str,
        *,
        interval_ms: int,
        compute_fn: ComputeFn,
        on_result: Optional[ResultFn] = None,
        schedule_type: str = "Continuous",
        due_fn: Optional[DueFn] = None,
        run_immediately: bool = True,
    ) -> None:
        if not tab_id:
            raise ValueError("tab_id required")
        if not self.is_running:
            self.start()
        interval_ms = max(100, int(interval_ms))
        with self._lock:
            existing = self._tabs.get(tab_id)
            if existing is not None:
                existing.stop = True
                task = existing.task
                loop = self._loop
            else:
                task = None
                loop = self._loop
            reg = _TabRegistration(
                tab_id=tab_id,
                interval_ms=interval_ms,
                compute_fn=compute_fn,
                on_result=on_result,
                schedule_type=schedule_type or "Continuous",
                due_fn=due_fn,
            )
            self._tabs[tab_id] = reg

        def _start_task():
            if task is not None and not task.done():
                task.cancel()
            reg.task = asyncio.create_task(
                self._tab_loop(reg, run_immediately=run_immediately),
                name=f"mon-{tab_id[:8]}",
            )

        assert loop is not None
        loop.call_soon_threadsafe(_start_task)
        logger.info(
            "Registered tab=%s interval_ms=%d schedule=%s",
            tab_id,
            interval_ms,
            schedule_type,
        )

    def unregister_tab(self, tab_id: str) -> None:
        with self._lock:
            reg = self._tabs.pop(tab_id, None)
            loop = self._loop
        if reg is None:
            return
        reg.stop = True

        def _cancel():
            if reg.task and not reg.task.done():
                reg.task.cancel()

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(_cancel)
        logger.info("Unregistered tab=%s", tab_id)

    def submit_once(
        self,
        tab_id: str,
        compute_fn: ComputeFn,
        on_result: Optional[ResultFn] = None,
        *,
        schedule_type: str = "On-Demand",
    ) -> bool:
        """Queue a single cycle (On-Demand / agent request)."""
        if not self.is_running:
            self.start()
        loop = self._loop
        executor = self._executor
        if loop is None or executor is None:
            return False

        with self._lock:
            reg = self._tabs.get(tab_id)

        async def _once():
            # Respect per-tab overlap policy when the tab is already registered.
            if reg is not None and reg.busy:
                metrics = CycleMetrics(
                    tab_id=tab_id,
                    started_at=time.time(),
                    status="skipped_overlap",
                    schedule_type=schedule_type,
                )
                self._record_metrics(metrics)
                logger.info("cycle tab=%s status=skipped_overlap total_ms=0", tab_id)
                return
            await self._run_cycle(
                tab_id=tab_id,
                compute_fn=compute_fn,
                on_result=on_result or (reg.on_result if reg else None),
                schedule_type=schedule_type,
                reg=reg,
            )

        loop.call_soon_threadsafe(lambda: asyncio.create_task(_once(), name=f"once-{tab_id[:8]}"))
        return True

    def get_recent_metrics(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._metrics)[-max(1, int(limit)) :]
        return [m.to_dict() for m in items]

    def active_tab_ids(self) -> list[str]:
        with self._lock:
            return list(self._tabs.keys())

    # ------------------------------------------------------------------ internals

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    async def _tab_loop(self, reg: _TabRegistration, *, run_immediately: bool) -> None:
        """Interval timer that does not await compute — overlap is skipped."""
        try:
            if run_immediately:
                self._schedule_tick(reg)
            while not reg.stop and not self._stop.is_set():
                await asyncio.sleep(reg.interval_ms / 1000.0)
                if reg.stop or self._stop.is_set():
                    break
                self._schedule_tick(reg)
        except asyncio.CancelledError:
            return

    def _schedule_tick(self, reg: _TabRegistration) -> None:
        """Fire a cycle without blocking the interval loop."""
        if reg.schedule_type == "Scheduled" and reg.due_fn is not None:
            try:
                if not reg.due_fn():
                    return
            except Exception as exc:
                logger.warning("due_fn failed for %s: %s", reg.tab_id, exc)
                return
        if reg.busy:
            metrics = CycleMetrics(
                tab_id=reg.tab_id,
                started_at=time.time(),
                status="skipped_overlap",
                schedule_type=reg.schedule_type,
            )
            self._record_metrics(metrics)
            logger.info(
                "cycle tab=%s status=skipped_overlap total_ms=0",
                reg.tab_id,
            )
            return
        # Claim the slot before scheduling so interval ticks cannot double-start.
        reg.busy = True
        asyncio.create_task(
            self._run_cycle(
                tab_id=reg.tab_id,
                compute_fn=reg.compute_fn,
                on_result=reg.on_result,
                schedule_type=reg.schedule_type,
                reg=reg,
                already_busy=True,
            ),
            name=f"cycle-{reg.tab_id[:8]}",
        )

    async def _run_cycle(
        self,
        *,
        tab_id: str,
        compute_fn: ComputeFn,
        on_result: Optional[ResultFn],
        schedule_type: str,
        reg: Optional[_TabRegistration],
        already_busy: bool = False,
    ) -> None:
        started = time.time()
        if reg is not None and not already_busy:
            if reg.busy:
                metrics = CycleMetrics(
                    tab_id=tab_id,
                    started_at=started,
                    status="skipped_overlap",
                    schedule_type=schedule_type,
                )
                self._record_metrics(metrics)
                logger.info("cycle tab=%s status=skipped_overlap total_ms=0", tab_id)
                return
            reg.busy = True
        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None

        queue_start = time.perf_counter()
        compute_ms = 0.0
        queue_wait_ms = 0.0
        status = "ok"
        error = ""
        result: Any = None
        try:
            compute_start = time.perf_counter()
            # queue_wait approximates time until executor picks up the job
            fut = loop.run_in_executor(executor, compute_fn)
            # Mark queue wait at submission; refine after await with executor lag estimate
            result = await fut
            compute_ms = (time.perf_counter() - compute_start) * 1000.0
            queue_wait_ms = max(0.0, (time.perf_counter() - queue_start) * 1000.0 - compute_ms)
            if isinstance(result, dict) and result.get("status") == "error":
                status = "error"
                error = str(result.get("error") or "")
            elif isinstance(result, dict) and result.get("status") == "skipped":
                status = "skipped"
        except Exception as exc:
            status = "error"
            error = str(exc)
            result = {"status": "error", "error": error, "log_lines": []}
            compute_ms = (time.perf_counter() - queue_start) * 1000.0
            queue_wait_ms = 0.0
            logger.error("Monitoring cycle failed tab=%s: %s", tab_id, exc)
        finally:
            if reg is not None:
                reg.busy = False

        total_ms = (time.time() - started) * 1000.0
        metrics = CycleMetrics(
            tab_id=tab_id,
            started_at=started,
            queue_wait_ms=round(queue_wait_ms, 2),
            compute_ms=round(compute_ms, 2),
            total_ms=round(total_ms, 2),
            status=status,
            error=error[:500],
            schedule_type=schedule_type,
        )
        self._record_metrics(metrics)
        logger.info(
            "cycle tab=%s status=%s compute_ms=%.2f queue_wait_ms=%.2f total_ms=%.2f",
            tab_id,
            status,
            metrics.compute_ms,
            metrics.queue_wait_ms,
            metrics.total_ms,
        )
        if on_result is not None:
            try:
                on_result(tab_id, result, metrics)
            except Exception as exc:
                logger.warning("on_result failed for %s: %s", tab_id, exc)

    def _record_metrics(self, metrics: CycleMetrics) -> None:
        with self._lock:
            self._metrics.append(metrics)


_PIPELINE: Optional[AsyncMonitoringPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_async_pipeline() -> AsyncMonitoringPipeline:
    """Process-wide singleton."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = AsyncMonitoringPipeline()
        return _PIPELINE


def reset_async_pipeline_for_tests() -> None:
    """Stop and clear singleton (unit tests only)."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            try:
                _PIPELINE.stop(wait=True)
            except Exception:
                pass
            _PIPELINE = None
