"""
Per-caller request rate limiting for the expensive assessment endpoints.

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

DECLARED LIMITATIONS
--------------------
1. Process-local, like core/circuit_breaker.py. Two replicas each enforce the
   configured rate, so the effective global limit is N x the configured value.
   A distributed limiter needs shared state — the Redis instance
   core/cache_backend.py already introduces is the natural home, and this
   module's `RateLimiter` interface is deliberately narrow enough to swap.
2. The bucket registry is LRU-capped (`RATE_LIMIT_MAX_TRACKED`) so that
   rotating identities cannot exhaust memory. The tradeoff is real and stated
   rather than hidden: a caller who can cycle through more than that many
   distinct identities can evict their own bucket and reset their budget.
   Bounded memory was judged the more important property, since memory
   exhaustion takes down every tenant while budget evasion degrades one limit.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from core.logger import get_logger

logger = get_logger(__name__)


class _Bucket:
    """One caller's token bucket. Not thread-safe on its own; the owning
    RateLimiter holds a lock around every mutation."""

    __slots__ = ("tokens", "last_refill")

    def __init__(self, tokens: float, now: float):
        self.tokens = tokens
        self.last_refill = now


class RateLimiter:
    """
    Token-bucket limiter keyed by an opaque identity string.

    `capacity` is the burst size; `refill_per_second` is the sustained rate.
    Both are supplied per-call rather than fixed at construction, because
    authenticated and anonymous callers get different allowances from the same
    limiter instance.
    """

    def __init__(self, name: str, max_tracked: int = 10_000):
        self.name = name
        self.max_tracked = max_tracked
        self._lock = threading.Lock()
        self._buckets: "OrderedDict[str, _Bucket]" = OrderedDict()

    def check(self, identity: str, capacity: float, refill_per_second: float):
        """
        Attempts to spend one token for `identity`.

        Returns (allowed, retry_after_seconds). `retry_after_seconds` is 0.0
        when allowed, and otherwise the time until one token is available —
        which is what a caller should put in a `Retry-After` header.

        A non-positive rate disables limiting for that caller rather than
        blocking everything, so a misconfiguration degrades to "no limit"
        loudly at config-validation time rather than silently to "total
        outage" at request time.
        """
        if refill_per_second <= 0 or capacity <= 0:
            return True, 0.0

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

    def _evict_if_needed(self):
        """Caller must hold the lock. Drops least-recently-used buckets."""
        while len(self._buckets) > self.max_tracked:
            evicted, _ = self._buckets.popitem(last=False)
            logger.debug(
                f"Rate limiter '{self.name}' evicted LRU bucket for "
                f"{evicted!r} (tracking cap {self.max_tracked})."
            )

    def reset(self):
        """Test/ops hook: forget every bucket."""
        with self._lock:
            self._buckets.clear()

    def __len__(self):
        with self._lock:
            return len(self._buckets)


def bucket_parameters(requests_per_minute: float, burst_seconds: float):
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
# is the entire point, exactly as with core/circuit_breaker.py's breakers.
assess_rate_limiter = RateLimiter("assess")
