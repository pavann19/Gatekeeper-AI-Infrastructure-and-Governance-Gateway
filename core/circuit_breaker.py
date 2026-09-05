"""
A circuit breaker for judge backends (Ollama, Llama Guard).

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
separate CircuitBreaker instances).

SHARED STATE
-----------------------
When `REDIS_URL` is set and reachable, breaker state (open timestamp +
consecutive-failure count) lives in Redis, so every Gatekeeper replica sees
one replica's discovery of a judge-backend outage immediately instead of
each re-discovering it independently. The half-open transition is an atomic
Lua check-and-clear: the first caller on any replica whose `is_open()` runs
after the cooldown clears the shared open-state once (not once per replica),
so a burst of blocked traffic resumes together rather than each replica
serving its own extra cooldown — the shared analogue of the process-local
"one state reset, then the next real call proceeds" behaviour.

If Redis is unset OR any Redis call fails at runtime, the breaker falls back
transparently to the original process-local behaviour — same pattern as
`core/rate_limit.py` and `core/cache_backend.py`.

READ-WITHOUT-SIDE-EFFECT DISCIPLINE
-----------------------------------
`core/metrics.refresh_circuit_breaker_gauges` and the `/health` check both
read `_opened_at` (under `_lock`) rather than calling `is_open()`, because
`is_open()` consumes the one-shot half-open probe slot. That contract is
preserved: `_opened_at` / `_consecutive_failures` are read-only views (Redis
GET when shared, local mirror otherwise) with NO half-open transition.
"""
import threading
import time

from core.logger import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "gk:cb:"


def _as_float(v):
    """redis-py returns bytes by default; normalise to float or None."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    return float(v)


def _as_int(v):
    if v is None:
        return 0
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    return int(v)

# is_open(): clear-and-pass exactly one caller when the cooldown has elapsed,
# otherwise report open. Uses Redis server TIME so replica clock skew is
# irrelevant. KEYS: [opened_at, fails]  ARGV: [cooldown_seconds]
_LUA_OPEN_CHECK = """
local opened = redis.call('GET', KEYS[1])
if not opened then return 0 end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
if now - tonumber(opened) >= tonumber(ARGV[1]) then
    redis.call('DEL', KEYS[1])
    redis.call('DEL', KEYS[2])
    return 0
end
return 1
"""

# record_failure(): INCR the failure counter; open the breaker iff the
# threshold is reached and it is not already open.
# KEYS: [opened_at, fails]  ARGV: [failure_threshold, ttl_seconds]
_LUA_RECORD_FAILURE = """
local fails = redis.call('INCR', KEYS[2])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
local opened = redis.call('GET', KEYS[1])
if (not opened) and fails >= tonumber(ARGV[1]) then
    local t = redis.call('TIME')
    local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
    redis.call('SET', KEYS[1], now)
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
    return {fails, 1}
end
return {fails, 0}
"""


class CircuitBreaker:
    def __init__(self, name, failure_threshold=3, cooldown_seconds=30, redis_client=None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        # Local mirror — authoritative when Redis is not in use, and the
        # fallback whenever a Redis call raises.
        self._local_opened_at = None
        self._local_fails = 0

        self._redis = redis_client
        self._k_opened = f"{_KEY_PREFIX}{name}:opened_at"
        self._k_fails = f"{_KEY_PREFIX}{name}:fails"
        # opened_at gets a TTL backstop so a crashed process cannot wedge the
        # breaker open forever; the is_open time-check clears it first in the
        # normal case.
        self._ttl = max(int(cooldown_seconds) * 4, 60)
        self._open_check = None
        self._record_failure = None
        if self._redis is not None:
            try:
                self._open_check = self._redis.register_script(_LUA_OPEN_CHECK)
                self._record_failure = self._redis.register_script(_LUA_RECORD_FAILURE)
            except Exception as e:  # noqa: BLE001
                logger.error("Circuit breaker '%s': cannot register Redis scripts "
                             "(%s: %s); using process-local state.",
                             name, type(e).__name__, e)
                self._redis = None

    # -- shared-state helpers ------------------------------------------------

    @property
    def _uses_redis(self):
        return self._redis is not None

    def _redis_peek_opened_at(self):
        """GET the open timestamp with NO half-open side effect. Returns a
        float, or None if closed. Raises on Redis error (caller falls back)."""
        return _as_float(self._redis.get(self._k_opened))

    # -- public API -------------------------------------------------------

    def is_open(self):
        """True means: do not attempt the call, fail fast instead."""
        if self._uses_redis:
            try:
                opened_before = _as_float(self._redis.get(self._k_opened))
                result = int(self._open_check(keys=[self._k_opened, self._k_fails],
                                              args=[self.cooldown_seconds]))
                if opened_before is not None and result == 0:
                    logger.info("Circuit breaker '%s' entering half-open probe (shared).",
                                self.name)
                self._local_opened_at = None if result == 0 else opened_before
                return bool(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Circuit breaker '%s': Redis is_open failed (%s: %s); "
                               "using local state.", self.name, type(e).__name__, e)

        with self._lock:
            if self._local_opened_at is None:
                return False
            if time.time() - self._local_opened_at >= self.cooldown_seconds:
                logger.info(f"Circuit breaker '{self.name}' entering half-open probe.")
                self._local_opened_at = None
                self._local_fails = 0
                return False
            return True

    def record_success(self):
        if self._uses_redis:
            try:
                self._redis.delete(self._k_opened, self._k_fails)
                self._local_opened_at = None
                self._local_fails = 0
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("Circuit breaker '%s': Redis record_success failed "
                               "(%s: %s); using local state.",
                               self.name, type(e).__name__, e)
        with self._lock:
            self._local_fails = 0
            self._local_opened_at = None

    def record_failure(self):
        """
        A single post-probe failure does not immediately re-open the breaker
        — it takes `failure_threshold` consecutive failures again, same as
        the original trip. This is deliberate: one flaky failure right after
        a cooldown (transient network blip) shouldn't force another full
        cooldown if the very next call would have succeeded. Sustained
        trouble still trips it again quickly, just not instantly on request 1.
        """
        if self._uses_redis:
            try:
                fails, opened = self._record_failure(
                    keys=[self._k_opened, self._k_fails],
                    args=[self.failure_threshold, self._ttl],
                )
                self._local_fails = _as_int(fails)
                if _as_int(opened) == 1:
                    self._local_opened_at = time.time()
                    logger.error(
                        "Circuit breaker '%s' OPEN after %s consecutive failures "
                        "(shared); failing fast for %ss.",
                        self.name, int(fails), self.cooldown_seconds,
                    )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("Circuit breaker '%s': Redis record_failure failed "
                               "(%s: %s); using local state.",
                               self.name, type(e).__name__, e)
        with self._lock:
            self._local_fails += 1
            if self._local_fails >= self.failure_threshold and self._local_opened_at is None:
                self._local_opened_at = time.time()
                logger.error(
                    f"Circuit breaker '{self.name}' OPEN after "
                    f"{self._local_fails} consecutive failures; "
                    f"failing fast for {self.cooldown_seconds}s."
                )

    def reset(self):
        """Test/ops hook: force the breaker fully closed."""
        if self._uses_redis:
            try:
                self._redis.delete(self._k_opened, self._k_fails)
            except Exception as e:  # noqa: BLE001
                logger.warning("Circuit breaker '%s': Redis reset failed (%s: %s).",
                               self.name, type(e).__name__, e)
        with self._lock:
            self._local_fails = 0
            self._local_opened_at = None

    # -- read-only views (NO half-open side effect) — used by metrics and
    #    /health, which must not consume the probe slot -----------------------

    @property
    def _opened_at(self):
        if self._uses_redis:
            try:
                v = self._redis_peek_opened_at()
                self._local_opened_at = v
                return v
            except Exception as e:  # noqa: BLE001
                logger.debug("Circuit breaker '%s': Redis peek failed (%s: %s); "
                             "reporting local mirror.", self.name, type(e).__name__, e)
        return self._local_opened_at

    @_opened_at.setter
    def _opened_at(self, value):
        # Retained so any legacy direct assignment keeps the local mirror
        # coherent; the Redis write path is the Lua scripts above.
        self._local_opened_at = value

    @property
    def _consecutive_failures(self):
        if self._uses_redis:
            try:
                return _as_int(self._redis.get(self._k_fails))
            except Exception:  # noqa: BLE001
                return self._local_fails
        return self._local_fails

    @_consecutive_failures.setter
    def _consecutive_failures(self, value):
        self._local_fails = value


def _build(name, failure_threshold=3, cooldown_seconds=30):
    """One breaker per judge backend. Shared via Redis when REDIS_URL is set
    and reachable (core.redis_client's pooled client), else process-local."""
    redis_client = None
    try:
        from core.redis_client import get_redis_client
        redis_client = get_redis_client()
    except Exception as e:  # noqa: BLE001
        logger.error("Circuit breaker '%s': redis client lookup failed (%s: %s); "
                     "using process-local state.", name, type(e).__name__, e)
    if redis_client is not None:
        logger.info("Circuit breaker '%s' state is shared across instances via Redis.", name)
    return CircuitBreaker(name, failure_threshold, cooldown_seconds, redis_client=redis_client)


# Module-level singletons — one breaker per judge backend, shared across all
# calls within this process (and, with Redis, across all replicas).
ollama_judge_breaker = _build("ollama_judge", failure_threshold=3, cooldown_seconds=30)
llama_guard_breaker = _build("llama_guard", failure_threshold=3, cooldown_seconds=30)
