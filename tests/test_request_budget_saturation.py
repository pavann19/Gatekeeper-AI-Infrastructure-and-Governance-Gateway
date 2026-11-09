import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from api.main import _submit_bounded
from core.request_budget import deadline


def test_timeout_keeps_capacity_until_worker_really_finishes():
    async def scenario():
        pool = ThreadPoolExecutor(max_workers=1)
        slots = threading.BoundedSemaphore(1)
        release = threading.Event()
        try:
            with pytest.raises(TimeoutError):
                await _submit_bounded(pool, slots, release.wait, (), 0.02)
            with pytest.raises(HTTPException) as failure:
                await _submit_bounded(pool, slots, lambda: None, (), 1)
            assert failure.value.status_code == 503
        finally:
            release.set()
            pool.shutdown(wait=True)
        assert slots.acquire(blocking=False)
    asyncio.run(scenario())


def test_cumulative_budget_is_not_reset_between_stages():
    async def scenario():
        pool = ThreadPoolExecutor(max_workers=1)
        slots = threading.BoundedSemaphore(2)
        token = deadline.set(time.monotonic() + 0.1)
        try:
            await _submit_bounded(pool, slots, time.sleep, (0.06,), 1)
            with pytest.raises(TimeoutError):
                await _submit_bounded(pool, slots, time.sleep, (0.06,), 1)
        finally:
            deadline.reset(token)
            pool.shutdown(wait=True)
    asyncio.run(scenario())


def test_cancelled_queued_work_does_not_execute():
    async def scenario():
        pool = ThreadPoolExecutor(max_workers=1)
        slots = threading.BoundedSemaphore(2)
        release = threading.Event()
        started = threading.Event()
        calls = []
        def block():
            started.set()
            release.wait()
        first = asyncio.create_task(_submit_bounded(pool, slots, block, (), 1))
        try:
            await asyncio.to_thread(started.wait)
            second = asyncio.create_task(_submit_bounded(pool, slots, lambda: calls.append(1), (), 1))
            await asyncio.sleep(0.01)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
        finally:
            release.set()
            await first
            pool.shutdown(wait=True)
        assert calls == []
    asyncio.run(scenario())
