"""One monotonic budget propagated into blocking workers."""
import time
from contextvars import ContextVar

deadline: ContextVar[float | None] = ContextVar('request_deadline', default=None)


def remaining(limit: float) -> float:
    end = deadline.get()
    value = min(limit, end - time.monotonic()) if end is not None else limit
    if value <= 0:
        raise TimeoutError("Request deadline exhausted")
    return value
