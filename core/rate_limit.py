"""
Per-caller request rate limiting for the expensive assessment and gateway endpoints.

WHY THIS EXISTS
---------------
Every `/api/v1/assess` call runs three transformer detectors and, in the
ambiguous zone, an LLM judge. Measured cold-path p95 is multiple seconds
(docs/ENGINEERING_ASSESSMENT.md §1p). With no limit, a single caller — or a
single broken retry loop — can saturate the bounded worker pool indefinitely
and deny service to every other tenant. This is the cheapest control that
turns "shared gateway" from an aspiration into a property.

WHY TOKEN BUCKET
----------------
A sliding-window log is more accurate but costs O(requests) memory per
caller, which is itself the DoS vector we are trying to close. A token bucket
is two floats per caller and permits a bounded burst — the right shape for an
API, where clients legitimately arrive in bursts and the thing worth
constraining is the SUSTAINED rate.

WHOSE BUCKET: THE PART THAT IS ACTUALLY A SECURITY DECISION
-----------------------------------------------------------
Authenticated callers are bucketed by `key_id`, which is resolved server-side
from a verified credential (core/auth.py) and therefore cannot be forged.

Anonymous callers have no such identity, and the two obvious options are both
wrong in different ways. One shared anonymous bucket is a self-inflicted
denial of service: a single abuser locks out every anonymous caller. Bucketing
by a client-supplied header is worse — anyone can mint unlimited identities by
rotating it.

So anonymous traffic is bucketed by the transport-level peer address, which
the caller cannot choose. Behind a reverse proxy every request appears to come
from the proxy, so `RATE_LIMIT_TRUST_FORWARDED_FOR` exists — but it is OFF by
default, because trusting `X-Forwarded-For` on a directly-exposed service
hands every caller an unlimited-identity bypass. When it is enabled, the
RIGHTMOST entry is used, not the leftmost: with one trusted proxy in front,
the rightmost value is the address that proxy actually observed, whereas the
leftmost is whatever the client put there. This assumes exactly one trusted
hop; deployments with a deeper chain must adjust it deliberately.

DISTRIBUTED VS LOCAL BACKENDS
-----------------------------
1. Redis-backed (`RedisRateLimiter`): Shared state across all Gatekeeper replicas.
   Executes an atomic Lua script for token-bucket refill and spend. Uses Redis
   server time (`redis.call('TIME')`) so small host clock differences between
   replicas never cause inconsistent refills. Expired keys are automatically
   evicted by Redis TTL, avoiding memory leaks.
2. Local fallback (`LocalRateLimiter` / `RateLimiter`): Process-local in-memory
   token bucket with LRU cap (`RATE_LIMIT_MAX_TRACKED`). Used when REDIS_URL
   is unset or when Redis experiences runtime connection failures.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

from core.logger import get_logger

logger = get_logger(__name__)


class _Bucket:
    """One caller's token bucket. Not thread-safe on its own; the owning
    LocalRateLimiter holds a lock around every mutation."""

    __slots__ = ("tokens", "last_refill")

    def __init__(self, tokens: float, now: float):
        self.tokens = tokens
        self.last_refill = now


class RateLimiterBackend:
    """Interface: token-bucket check and state management."""

    def check(
        self,
        identity: str,
        capacity: float,
        refill_per_second: float,
        now: float | None = None,
    ) -> tuple[bool, float]:
        """
        Attempts to spend one token for `identity`.

        Returns (allowed, retry_after_seconds). `retry_after_seconds` is 0.0
        when allowed, and otherwise the time until one token is available —
        which is what a caller should put in a `Retry-After` header.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Test/ops hook: forget every bucket."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of tracked identities/keys."""
        raise NotImplementedError


class LocalRateLimiter(RateLimiterBackend):
    """
    Process-local token-bucket limiter keyed by an opaque identity string.

    `capacity` is the burst size; `refill_per_second` is the sustained rate.
    Both are supplied per-call rather than fixed at construction, because
    authenticated and anonymous callers get different allowances from the same
    limiter instance.
    """

    def __init__(self, name: str, max_tracked: int = 10_000):
        self.name = name
        self.max_tracked = max_tracked
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def check(
        self,
        identity: str,
        capacity: float,
        refill_per_second: float,
        now: float | None = None,
    ) -> tuple[bool, float]:
        """
        Attempts to spend one token for `identity`.

        A non-positive rate disables limiting for that caller rather than
        blocking everything, so a misconfiguration degrades to "no limit"
        loudly at config-validation time rather than silently to "total
        outage" at request time.
        """
        if refill_per_second <= 0 or capacity <= 0:
            return True, 0.0

        if now is None:
            # time.monotonic, not time.time: a wall-clock adjustment (NTP step,
            # DST, manual change) must not hand out free tokens or freeze a bucket.
            now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=capacity, now=now)
                self._buckets[identity] = bucket
                self._evict_if_needed()
            else:
                self._buckets.move_to_end(identity)
                elapsed = now - bucket.last_refill
                if elapsed > 0:
                    bucket.tokens = min(capacity, bucket.tokens + elapsed * refill_per_second)
                    bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            deficit = 1.0 - bucket.tokens
            return False, deficit / refill_per_second

    def _evict_if_needed(self) -> None:
        """Caller must hold the lock. Drops least-recently-used buckets."""
        while len(self._buckets) > self.max_tracked:
            evicted, _ = self._buckets.popitem(last=False)
            logger.debug(
                f"Rate limiter '{self.name}' evicted LRU bucket for "
                f"{evicted!r} (tracking cap {self.max_tracked})."
            )

    def reset(self) -> None:
        """Test/ops hook: forget every bucket."""
        with self._lock:
            self._buckets.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)


# Backwards compatibility alias:
RateLimiter = LocalRateLimiter


LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_second = tonumber(ARGV[2])

if not capacity or not refill_per_second or capacity <= 0 or refill_per_second <= 0 then
    return {1, "0.0"}
end

local now
if ARGV[3] and ARGV[3] ~= "" and tonumber(ARGV[3]) and tonumber(ARGV[3]) > 0 then
    now = tonumber(ARGV[3])
else
    local t = redis.call('TIME')
    now = tonumber(t[1]) + (tonumber(t[2]) / 1000000.0)
end

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if not tokens or not last_refill then
    tokens = capacity
    last_refill = now
else
    local elapsed = now - last_refill
    if elapsed > 0 then
        tokens = math.min(capacity, tokens + (elapsed * refill_per_second))
        last_refill = now
    end
end

local allowed = 0
local retry_after = 0.0

if tokens >= 1.0 then
    tokens = tokens - 1.0
    allowed = 1
    retry_after = 0.0
else
    allowed = 0
    local deficit = 1.0 - tokens
    retry_after = deficit / refill_per_second
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))

-- Auto-expire key after it is fully refilled + safety buffer to avoid leaking Redis memory
local fill_time = math.ceil(capacity / refill_per_second)
local ttl = math.max(3600, fill_time * 2)
redis.call('EXPIRE', key, ttl)

return {allowed, tostring(retry_after)}
"""


class RedisRateLimiter(RateLimiterBackend):
    """
    Shared distributed token-bucket limiter backed by Redis.
    Uses an atomic Lua script to synchronize tokens and timestamps across
    all Gatekeeper replicas. Falls back cleanly to an internal LocalRateLimiter
    if Redis operations encounter runtime errors.
    """

    KEY_PREFIX = "gatekeeper:ratelimit:"

    def __init__(self, client, name: str, max_tracked: int = 10_000):
        self._client = client
        self.name = name
        self.max_tracked = max_tracked
        self._local_fallback = LocalRateLimiter(name=name, max_tracked=max_tracked)
        try:
            self._script = self._client.register_script(LUA_TOKEN_BUCKET)
        except Exception:
            self._script = None

    def _redis_key(self, identity: str) -> str:
        return f"{self.KEY_PREFIX}{self.name}:{identity}"

    def check(
        self,
        identity: str,
        capacity: float,
        refill_per_second: float,
        now: float | None = None,
    ) -> tuple[bool, float]:
        if refill_per_second <= 0 or capacity <= 0:
            return True, 0.0

        key = self._redis_key(identity)
        now_arg = str(now) if now is not None else ""

        try:
            if self._script is not None:
                res = self._script(keys=[key], args=[str(capacity), str(refill_per_second), now_arg])
            else:
                res = self._client.eval(LUA_TOKEN_BUCKET, 1, key, str(capacity), str(refill_per_second), now_arg)

            allowed_raw = res[0]
            retry_after_raw = res[1]
            if isinstance(retry_after_raw, bytes):
                retry_after_raw = retry_after_raw.decode("utf-8")
            allowed = bool(int(allowed_raw))
            retry_after = float(retry_after_raw)
            return allowed, retry_after
        except Exception as e:
            logger.error(
                f"Redis rate limiter check failed ({type(e).__name__}: {e}); "
                f"falling back to local in-memory limiter for identity {identity!r}."
            )
            return self._local_fallback.check(identity, capacity, refill_per_second, now=now)

    def reset(self) -> None:
        """Deletes all keys belonging to this rate limiter instance in Redis and resets local fallback."""
        self._local_fallback.reset()
        pattern = f"{self.KEY_PREFIX}{self.name}:*"
        try:
            for k in self._client.scan_iter(pattern):
                self._client.delete(k)
        except Exception as e:
            logger.error(f"Redis rate limiter reset failed ({type(e).__name__}: {e}).")

    def __len__(self) -> int:
        """Returns the count of active keys tracked in Redis for this limiter instance."""
        pattern = f"{self.KEY_PREFIX}{self.name}:*"
        try:
            count = 0
            for _ in self._client.scan_iter(pattern):
                count += 1
            return count
        except Exception as e:
            logger.error(f"Redis rate limiter __len__ failed ({type(e).__name__}: {e}).")
            return len(self._local_fallback)


def build_rate_limiter(name: str = "assess", client=None, max_tracked: int = 10_000) -> RateLimiterBackend:
    """
    Selects Redis if `client` is provided or if `REDIS_URL` is set AND reachable,
    using the shared ConnectionPool in `core.redis_client`.
    Falls back gracefully to the local in-memory limiter.
    """
    if client is not None:
        return RedisRateLimiter(client, name=name, max_tracked=max_tracked)

    from core.redis_client import get_redis_client

    redis_client = get_redis_client()
    if redis_client is not None:
        logger.info(f"Rate limiter '{name}' is now shared across instances via Redis.")
        return RedisRateLimiter(redis_client, name=name, max_tracked=max_tracked)

    if os.environ.get("REDIS_URL"):
        logger.error(
            f"REDIS_URL is set but Redis is unreachable; falling back to local "
            f"in-memory rate limiter for '{name}'. Rate limits will NOT be shared "
            f"across gateway instances until this is fixed."
        )
    else:
        logger.info(f"REDIS_URL not set; using local in-memory rate limiter for '{name}' (single-node only).")

    return LocalRateLimiter(name=name, max_tracked=max_tracked)




def bucket_parameters(requests_per_minute: float, burst_seconds: float) -> tuple[float, float]:
    """
    Converts a human-facing "N requests per minute" limit into the
    (capacity, refill_per_second) pair the bucket actually needs.

    Capacity is floored at one token: a limit low enough that the burst
    rounds to zero should still let a single request through eventually,
    rather than rejecting a caller who is within their sustained rate.
    """
    refill_per_second = requests_per_minute / 60.0
    capacity = max(1.0, refill_per_second * burst_seconds)
    return capacity, refill_per_second


# Module-level singleton, shared across all requests in this process — which
# is distributed if REDIS_URL is set, and process-local if not.
assess_rate_limiter = build_rate_limiter("assess")

# A SEPARATE singleton for the MCP HTTP/SSE transport (core/mcp_http_server.py)
# -- a distinct process from the main API (its own uvicorn instance, its own
# port), so sharing `assess_rate_limiter`'s bucket namespace would be a real
# bug once REDIS_URL is set: the whole point of the Redis backend is to unify
# state across independent PROCESSES that are meant to share one budget (the
# main API's own replicas), not to unify state across independent SERVICES
# with a caller who happens to reuse the same API key on both. Same reasoning
# `_gateway_pool` being separate from `_assess_pool` in api/main.py already
# documents for an analogous case -- the wrong coupling between traffic
# classes, not just a wrong coupling between requests.
mcp_rate_limiter = build_rate_limiter("mcp")
