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

WHY A DAILY WINDOW, PROCESS-LOCAL, MIRRORING core/rate_limit.py
-----------------------------------------------------------------
Same declared limitations as RateLimiter: two replicas each enforce the
configured quota independently, so the effective global quota is N x the
configured value in a multi-process deployment (fixable later by moving the
counter to the shared Redis backend core/cache_backend.py already
introduces, same note core/rate_limit.py makes about itself). The registry
is LRU-capped for the same reason RateLimiter's bucket dict is: an unbounded
number of distinct tenant_ids must not be a memory-exhaustion vector, though
in practice tenant_id is operator-provisioned (core/tenancy.py), not
caller-chosen, so this is a defensive bound rather than an anticipated attack.

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


class _TenantUsage:
    __slots__ = ("day", "tokens_used")

    def __init__(self, day: str):
        self.day = day
        self.tokens_used = 0


class TokenQuotaTracker:
    """
    Tracks cumulative token usage per tenant per UTC calendar day.

    `max_tracked` bounds memory the same way RateLimiter._buckets does — see
    module docstring's "defensive bound rather than an anticipated attack".
    """

    def __init__(self, max_tracked: int = 10_000):
        self.max_tracked = max_tracked
        self._lock = threading.Lock()
        self._usage: "OrderedDict[str, _TenantUsage]" = OrderedDict()

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

    def _evict_if_needed(self):
        while len(self._usage) > self.max_tracked:
            evicted, _ = self._usage.popitem(last=False)
            logger.debug(
                f"Token quota tracker evicted LRU entry for tenant "
                f"{evicted!r} (tracking cap {self.max_tracked})."
            )

    def reset(self):
        """Test/ops hook: forget every tenant's tracked usage."""
        with self._lock:
            self._usage.clear()


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


# Module-level singleton, shared across all requests in this process — same
# reasoning as core/rate_limit.py's assess_rate_limiter.
gateway_token_quota = TokenQuotaTracker()
