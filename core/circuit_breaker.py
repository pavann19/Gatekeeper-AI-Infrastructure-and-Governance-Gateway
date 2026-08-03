"""
A small in-process circuit breaker for judge backends (Ollama, Llama Guard).

WHY THIS EXISTS: before this, a slow or failing judge backend made EVERY
ambiguous-zone request pay the full cost of trying it, individually — Ollama's
`requests.post(..., timeout=30)`, or Llama Guard's multi-second model call
that can itself hang under memory pressure. Under sustained backend trouble
this cascades: every request landing in the ambiguous zone queues behind the
same slow failure, one at a time, for as long as the backend stays broken.
The async fix (core/risk.py's background scheduler) does not help here — the
FAST path still calls judge_arbitration synchronously, and that is exactly
the call this wraps.

A circuit breaker trips after a run of consecutive failures for a given
backend and then fails FAST — without attempting the call at all — for a
cooldown window, letting the pipeline's existing fail-closed defaults take
over immediately instead of every request re-discovering the same outage the
slow way. After the cooldown, exactly one request is let through as a probe;
if it succeeds the breaker resets, if it fails the cooldown starts again.

Deliberately per-backend (Ollama and Llama Guard are tracked separately, via
separate CircuitBreaker instances) and process-local — a distributed breaker
needs shared state (the same Redis instance core/cache_backend.py already
introduces would be the natural place), which is future work. A per-instance
breaker still stops a single instance from hammering a backend that is
already known-down, which is most of the practical value.
"""
import threading
import time

from core.logger import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    def __init__(self, name, failure_threshold=3, cooldown_seconds=30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at = None

    def is_open(self):
        """True means: do not attempt the call, fail fast instead."""
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at >= self.cooldown_seconds:
                # Half-open: let exactly one probe through by resetting state
                # now. If that probe fails, record_failure() re-opens it
                # immediately (threshold is already at 0, so failure #1 after
                # a probe re-trips on the NEXT failure, not this one alone --
                # see the docstring on record_failure for why that's fine).
                logger.info(f"Circuit breaker '{self.name}' entering half-open probe.")
                self._opened_at = None
                self._consecutive_failures = 0
                return False
            return True

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self):
        """
        A single post-probe failure does not immediately re-open the breaker
        — it takes `failure_threshold` consecutive failures again, same as
        the original trip. This is deliberate: one flaky failure right after
        a cooldown (transient network blip) shouldn't force another full
        cooldown if the very next call would have succeeded. Sustained
        trouble still trips it again quickly, just not instantly on request 1.
        """
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold and self._opened_at is None:
                self._opened_at = time.time()
                logger.error(
                    f"Circuit breaker '{self.name}' OPEN after "
                    f"{self._consecutive_failures} consecutive failures; "
                    f"failing fast for {self.cooldown_seconds}s."
                )

    def reset(self):
        """Test/ops hook: force the breaker fully closed."""
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None


# Module-level singletons — one breaker per judge backend, shared across all
# calls within this process (not per-request), which is the whole point.
ollama_judge_breaker = CircuitBreaker("ollama_judge", failure_threshold=3, cooldown_seconds=30)
llama_guard_breaker = CircuitBreaker("llama_guard", failure_threshold=3, cooldown_seconds=30)
