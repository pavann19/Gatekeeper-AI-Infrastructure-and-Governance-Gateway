"""
Per-tenant daily token accounting for the LLM Gateway (Phase 5: "Token
accounting" roadmap item).

WHY THIS ENFORCES PAST THE FACT, NOT AHEAD OF IT
-------------------------------------------------
Token usage for a call is only known once the provider has answered — there
is no way to ask "how many tokens will this cost" before making the call.
So this tracker can only ever reject the NEXT call once a tenant's already-
recorded usage has crossed its quota; the call that pushes a tenant over the
line is always allowed to complete. This is the same shape every commercial
LLM API's usage cap actually has, not a limitation specific to this gateway.

DISTRIBUTED VS LOCAL BACKENDS
-----------------------------
1. Redis-backed (`RedisTokenQuotaTracker`): Shared state across all replicas.
   Keys are per-tenant per-UTC-calendar-day (`gatekeeper:quota:<tenant_id>:<YYYY-MM-DD>`).
   Uses atomic Redis increment operations (`INCRBY` / Lua script) with automatic
   midnight-rollover TTL.
2. Local fallback (`LocalTokenQuotaTracker` / `TokenQuotaTracker`): In-memory
   thread-safe tracker with LRU cap. Used when REDIS_URL is unset or when Redis
   is temporarily unavailable.

WHY A DAILY WINDOW, PROCESS-LOCAL AS FALLBACK, MIRRORING core/rate_limit.py
---------------------------------------------------------------------------
The local fallback uses the same bounded in-memory strategy as RateLimiter:
an LRU-capped registry prevents unbounded tenant_id proliferation from becoming
a memory exhaustion vector (though tenant_id is operator-provisioned via
core/tenancy.py, this provides defensive bounding). In multi-replica deployments,
RedisTokenQuotaTracker ensures exact global quota enforcement across all nodes.

UNMETERED PROVIDERS AND USAGE SHAPES
------------------------------------
Usage that a provider does not report (Ollama's /api/chat returns none) adds
zero to the tracked total for that call — there is nothing to count, and
silently estimating a number would misrepresent actual spend as a metered
fact. A tenant calling only Ollama will never be quota-limited by this
module no matter how much they use it; that is a stated gap, not a bug, and
sits alongside `LLMResponse.usage`'s own "not consumed by anything" note
in core/llm_providers.py that this module is now the one place that DOES
consume it.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone

from core.logger import get_logger

logger = get_logger(__name__)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def seconds_until_utc_midnight() -> float:
    """What a caller should put in a Retry-After header when quota-blocked —
    the window resets at UTC midnight, not "some arbitrary interval later"."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    next_midnight = datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    return max(0.0, (next_midnight - now).total_seconds())


class TokenQuotaBackend:
    """Interface for token quota tracking backends."""

    def would_exceed(self, tenant_id: str, quota: int) -> bool:
        raise NotImplementedError

    def usage_today(self, tenant_id: str) -> int:
        raise NotImplementedError

    def record(self, tenant_id: str, tokens: int) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class _TenantUsage:
    __slots__ = ("day", "tokens_used")

    def __init__(self, day: str):
        self.day = day
        self.tokens_used = 0


class LocalTokenQuotaTracker(TokenQuotaBackend):
    """
    Tracks cumulative token usage per tenant per UTC calendar day in memory.
    `max_tracked` bounds memory via LRU eviction.
    """

    def __init__(self, max_tracked: int = 10_000):
        self.max_tracked = max_tracked
        self._lock = threading.Lock()
        self._usage: OrderedDict[str, _TenantUsage] = OrderedDict()

    def _get_current(self, tenant_id: str) -> _TenantUsage:
        """Caller must hold the lock. Rolls a tenant's counter over to a
        fresh day boundary transparently — a request at 00:00:01 UTC must
        not still be charged against yesterday's total."""
        today = _today_utc()
        usage = self._usage.get(tenant_id)
        if usage is None or usage.day != today:
            usage = _TenantUsage(today)
            self._usage[tenant_id] = usage
            self._evict_if_needed()
        else:
            self._usage.move_to_end(tenant_id)
        return usage

    def would_exceed(self, tenant_id: str, quota: int) -> bool:
        """
        True if this tenant has already used up their daily quota — the call
        being considered has not happened yet and adds nothing here; see
        `record()` for after a call completes.

        quota <= 0 means unlimited: always returns False.
        """
        if quota <= 0:
            return False
        with self._lock:
            usage = self._get_current(tenant_id)
            return usage.tokens_used >= quota

    def usage_today(self, tenant_id: str) -> int:
        with self._lock:
            return self._get_current(tenant_id).tokens_used

    def record(self, tenant_id: str, tokens: int) -> None:
        """Adds `tokens` to today's running total. A non-positive value is a
        no-op — nothing to record for a provider that reported no usage."""
        if tokens <= 0:
            return
        with self._lock:
            usage = self._get_current(tenant_id)
            usage.tokens_used += tokens

    def _evict_if_needed(self) -> None:
        while len(self._usage) > self.max_tracked:
            evicted, _ = self._usage.popitem(last=False)
            logger.debug(
                f"Token quota tracker evicted LRU entry for tenant "
                f"{evicted!r} (tracking cap {self.max_tracked})."
            )

    def reset(self) -> None:
        """Test/ops hook: forget every tenant's tracked usage."""
        with self._lock:
            self._usage.clear()


# Backwards compatibility alias:
TokenQuotaTracker = LocalTokenQuotaTracker


LUA_RECORD_QUOTA = """
local key = KEYS[1]
local tokens = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

local current = redis.call('INCRBY', key, tokens)
if current == tokens then
    redis.call('EXPIRE', key, ttl)
end
return current
"""


class RedisTokenQuotaTracker(TokenQuotaBackend):
    """
    Distributed daily token quota tracker backed by Redis.
    Falls back gracefully to an internal LocalTokenQuotaTracker on Redis errors.
    """

    KEY_PREFIX = "gatekeeper:quota:"

    def __init__(self, client, max_tracked: int = 10_000):
        self._client = client
        self.max_tracked = max_tracked
        self._local_fallback = LocalTokenQuotaTracker(max_tracked=max_tracked)
        try:
            self._record_script = self._client.register_script(LUA_RECORD_QUOTA)
        except Exception:
            self._record_script = None

    def _redis_key(self, tenant_id: str) -> str:
        return f"{self.KEY_PREFIX}{tenant_id}:{_today_utc()}"

    def would_exceed(self, tenant_id: str, quota: int) -> bool:
        if quota <= 0:
            return False

        key = self._redis_key(tenant_id)
        try:
            raw = self._client.get(key)
            used = int(raw) if raw is not None else 0
            return used >= quota
        except Exception as e:
            logger.error(
                f"Redis token quota would_exceed check failed ({type(e).__name__}: {e}); "
                f"falling back to local tracker for tenant {tenant_id!r}."
            )
            return self._local_fallback.would_exceed(tenant_id, quota)

    def usage_today(self, tenant_id: str) -> int:
        key = self._redis_key(tenant_id)
        try:
            raw = self._client.get(key)
            return int(raw) if raw is not None else 0
        except Exception as e:
            logger.error(
                f"Redis token quota usage_today check failed ({type(e).__name__}: {e}); "
                f"falling back to local tracker for tenant {tenant_id!r}."
            )
            return self._local_fallback.usage_today(tenant_id)

    def record(self, tenant_id: str, tokens: int) -> None:
        if tokens <= 0:
            return

        key = self._redis_key(tenant_id)
        # Keep until midnight plus 24 hours buffer so historical queries near boundary don't immediately lose data
        ttl = int(seconds_until_utc_midnight()) + 86400

        try:
            if self._record_script is not None:
                self._record_script(keys=[key], args=[str(tokens), str(ttl)])
            else:
                self._client.eval(LUA_RECORD_QUOTA, 1, key, str(tokens), str(ttl))
        except Exception as e:
            logger.error(
                f"Redis token quota record failed ({type(e).__name__}: {e}); "
                f"falling back to local tracker for tenant {tenant_id!r}."
            )
            self._local_fallback.record(tenant_id, tokens)

    def reset(self) -> None:
        self._local_fallback.reset()
        pattern = f"{self.KEY_PREFIX}*"
        try:
            for k in self._client.scan_iter(pattern):
                self._client.delete(k)
        except Exception as e:
            logger.error(f"Redis token quota reset failed ({type(e).__name__}: {e}).")


def build_token_quota_tracker(client=None, max_tracked: int = 10_000) -> TokenQuotaBackend:
    """
    Selects Redis if `client` is provided or if `REDIS_URL` is set AND reachable,
    using the shared ConnectionPool in `core.redis_client`.
    Falls back gracefully to the local in-memory tracker.
    """
    if client is not None:
        return RedisTokenQuotaTracker(client, max_tracked=max_tracked)

    from core.redis_client import get_redis_client

    redis_client = get_redis_client()
    if redis_client is not None:
        return RedisTokenQuotaTracker(redis_client, max_tracked=max_tracked)

    if os.environ.get("REDIS_URL"):
        logger.error("falling back to local in-memory token quota tracker.")

    return LocalTokenQuotaTracker(max_tracked=max_tracked)




def extract_total_tokens(usage) -> int:
    """
    Normalises the two usage shapes core/llm_providers.py's backends
    actually return into one integer, so the tracker never needs to know
    which provider produced a given LLMResponse.

    OpenAI-compatible: {"total_tokens": N, ...}. Anthropic-compatible:
    {"input_tokens": N, "output_tokens": M} with no combined field. Ollama:
    None (not reported at all). Anything else unrecognised counts as 0
    rather than raising — a malformed or future usage shape must degrade
    accounting, never a request in flight, matching this project's metrics
    tolerance convention (core/metrics.py's record_assessment docstring).
    """
    if not isinstance(usage, dict):
        return 0
    if isinstance(usage.get("total_tokens"), int):
        return usage["total_tokens"]
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if isinstance(input_tokens, int) or isinstance(output_tokens, int):
        return (input_tokens or 0) + (output_tokens or 0)
    return 0


# Module-level singleton, shared across all requests in this process — which
# is distributed if REDIS_URL is set, and process-local if not.
gateway_token_quota = build_token_quota_tracker()
