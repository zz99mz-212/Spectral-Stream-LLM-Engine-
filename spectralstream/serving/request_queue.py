from __future__ import annotations

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class QueuedRequest:
    request_id: str
    priority: int
    created_at: float
    payload: dict
    endpoint: str
    timeout_seconds: float = 300.0
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = field(default=None, compare=False, repr=False)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age_seconds > self.timeout_seconds

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class RequestQueue:
    def __init__(
        self,
        max_concurrent: int = 8,
        max_queued: int = 256,
        default_timeout: float = 300.0,
        batch_size: int = 1,
        batch_timeout_ms: float = 50.0,
    ):
        self._queue: list[QueuedRequest] = []
        self._max_queued = max_queued
        self._default_timeout = default_timeout
        self._batch_size = batch_size
        self._batch_timeout_ms = batch_timeout_ms

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_timeout = 0
        self._total_rejected = 0
        self._wait_times: list[float] = []
        self._active_requests: dict[str, QueuedRequest] = {}

    async def enqueue(
        self,
        payload: dict,
        endpoint: str = "chat",
        priority: int = Priority.NORMAL,
        timeout: Optional[float] = None,
    ) -> QueuedRequest:
        timeout = timeout or self._default_timeout

        async with self._lock:
            if len(self._queue) >= self._max_queued:
                self._total_rejected += 1
                raise asyncio.QueueFull(
                    f"Queue full: {len(self._queue)}/{self._max_queued}"
                )

            loop = asyncio.get_event_loop()
            future = loop.create_future()

            req = QueuedRequest(
                request_id=f"req_{uuid.uuid4().hex[:12]}",
                priority=priority,
                created_at=time.time(),
                payload=payload,
                endpoint=endpoint,
                timeout_seconds=timeout,
                future=future,
            )

            heapq.heappush(self._queue, req)
            self._total_enqueued += 1
            logger.debug(
                "request_enqueued id=%s endpoint=%s priority=%d queue_size=%d",
                req.request_id,
                endpoint,
                priority,
                len(self._queue),
            )
            return req

    async def dequeue(self) -> Optional[QueuedRequest]:
        async with self._lock:
            while self._queue:
                req = self._queue[0]
                if req.is_expired:
                    heapq.heappop(self._queue)
                    self._total_timeout += 1
                    if req.future and not req.future.done():
                        req.future.set_exception(
                            asyncio.TimeoutError(
                                f"Request {req.request_id} expired after {req.age_seconds:.1f}s"
                            )
                        )
                    continue
                heapq.heappop(self._queue)
                self._total_dequeued += 1
                self._wait_times.append(req.age_seconds)
                self._active_requests[req.request_id] = req
                return req
        return None

    async def process_next(self, handler: Callable) -> Optional[Any]:
        req = await self.dequeue()
        if req is None:
            return None

        async with self._semaphore:
            try:
                result = await handler(req)
                if req.future and not req.future.done():
                    req.future.set_result(result)
                return result
            except Exception as exc:
                if req.future and not req.future.done():
                    req.future.set_exception(exc)
                raise
            finally:
                self._active_requests.pop(req.request_id, None)

    async def cancel(self, request_id: str) -> bool:
        async with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    if req.future and not req.future.done():
                        req.future.set_exception(asyncio.CancelledError())
                    return True
        return False

    def peek(self) -> Optional[QueuedRequest]:
        return self._queue[0] if self._queue else None

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._active_requests)

    def stats(self) -> dict:
        avg_wait = sum(self._wait_times[-100:]) / max(len(self._wait_times[-100:]), 1)
        return {
            "queue_size": len(self._queue),
            "active_requests": len(self._active_requests),
            "max_queued": self._max_queued,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_timeout": self._total_timeout,
            "total_rejected": self._total_rejected,
            "avg_wait_ms": round(avg_wait * 1000, 2),
        }
