"""
Additional edge-case coverage for core/rate_limit.py, deliberately not
duplicating tests/test_rate_limit.py.

Focus areas not already exercised there:
  - Eviction of an identity actually RESETS its budget (the documented
    accepted limitation in core/rate_limit.py's module docstring, proven
    end-to-end rather than assumed).
  - Exact boundary behaviour at fractional capacities.
  - Bulk eviction when the registry is shrunk or overshoots by more than one.
  - `move_to_end` semantics: touching a bucket protects it from eviction,
    and NOT touching it does not.
  - Concurrent access across many distinct identities (registry-level
    thread-safety, not just single-bucket token accounting).
"""
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.rate_limit import RateLimiter


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr("core.rate_limit.time.monotonic", c)
    return c


# ---------------------------------------------------------------------------
# Eviction actually resets budget - the documented tradeoff, proven live.
# ---------------------------------------------------------------------------

def test_evicted_identity_gets_a_fresh_budget_on_return(clock):
    """
    The module docstring states that a caller who cycles through more than
    `max_tracked` distinct identities can evict their own bucket and reset
    their budget. This proves that actually happens, not just that eviction
    occurs.
    """
    limiter = RateLimiter("test", max_tracked=3)

    # 'victim' spends its entire budget.
    for _ in range(2):
        assert limiter.check("victim", capacity=2, refill_per_second=0.0001)[0] is True
    assert limiter.check("victim", capacity=2, refill_per_second=0.0001)[0] is False

    # Fill the registry with other identities without ever touching 'victim'
    # again, pushing it out as the least-recently-used entry.
    limiter.check("a", capacity=2, refill_per_second=0.0001)
    limiter.check("b", capacity=2, refill_per_second=0.0001)
    limiter.check("c", capacity=2, refill_per_second=0.0001)

    # Registry capacity is 3; 'victim' plus a/b/c is 4 distinct identities,
    # so the oldest untouched one ('victim') must have been evicted.
    assert len(limiter) == 3

    # Its budget must now be fully restored, not still exhausted - this is
    # the actual security-relevant consequence of the LRU cap.
    assert limiter.check("victim", capacity=2, refill_per_second=0.0001)[0] is True
    assert limiter.check("victim", capacity=2, refill_per_second=0.0001)[0] is True
    assert limiter.check("victim", capacity=2, refill_per_second=0.0001)[0] is False


def test_touching_a_bucket_protects_it_from_eviction(clock):
    """The inverse of the above: staying warm (being re-checked) must save
    an identity from eviction, proving eviction order is really LRU and not
    e.g. insertion order or random."""
    limiter = RateLimiter("test", max_tracked=2)

    limiter.check("warm", capacity=5, refill_per_second=0.0001)
    limiter.check("cold", capacity=5, refill_per_second=0.0001)

    # Keep 'warm' alive while pushing new identities through.
    for i in range(5):
        limiter.check("warm", capacity=5, refill_per_second=0.0001)
        limiter.check(f"churn:{i}", capacity=5, refill_per_second=0.0001)

    assert len(limiter) == 2
    # 'warm' must have survived every eviction round; 'cold' must not have.
    # We can't introspect membership directly, but we can prove 'warm' kept
    # accumulating spends against the SAME bucket by draining it exactly to
    # its known capacity of 5 (already spent 6 times above -> exhausted).
    assert limiter.check("warm", capacity=5, refill_per_second=0.0001)[0] is False


def test_bulk_overflow_evicts_down_to_the_cap_in_one_pass(clock):
    """If more than one identity over the cap is inserted, eviction must
    still converge to exactly max_tracked (not off-by-one, not left over)."""
    limiter = RateLimiter("test", max_tracked=5)

    for i in range(50):
        limiter.check(f"id:{i}", capacity=1, refill_per_second=1.0)

    assert len(limiter) == 5


# ---------------------------------------------------------------------------
# Exact boundary behaviour, including fractional capacity.
# ---------------------------------------------------------------------------

def test_fractional_capacity_allows_floor_of_capacity_requests(clock):
    """A capacity of 2.9 should behave like 2 whole tokens available - the
    2nd request allowed, the 3rd rejected, in the same window."""
    limiter = RateLimiter("test")

    assert limiter.check("id", capacity=2.9, refill_per_second=0.0001)[0] is True
    assert limiter.check("id", capacity=2.9, refill_per_second=0.0001)[0] is True
    allowed, _ = limiter.check("id", capacity=2.9, refill_per_second=0.0001)
    assert allowed is False


def test_nth_request_allowed_n_plus_1th_rejected_same_window(clock):
    """Explicit boundary check at N=7: exactly 7 requests succeed, the 8th
    fails, all within the same (non-advancing) window."""
    limiter = RateLimiter("test")
    n = 7

    results = [limiter.check("id", capacity=n, refill_per_second=0.0001)[0] for _ in range(n)]
    assert results == [True] * n

    allowed, _ = limiter.check("id", capacity=n, refill_per_second=0.0001)
    assert allowed is False


def test_boundary_with_positive_refill_rate(clock):
    """Same boundary check, but with a real positive refill rate so
    retry_after is computed (not short-circuited by the disable path)."""
    limiter = RateLimiter("test")
    n = 4

    for _ in range(n):
        assert limiter.check("id", capacity=n, refill_per_second=2.0)[0] is True

    allowed, retry_after = limiter.check("id", capacity=n, refill_per_second=2.0)
    assert allowed is False
    assert retry_after == pytest.approx(0.5)  # 1 token needed / 2 tokens-per-sec

    # Advancing exactly that long must flip it back to allowed, and only once.
    clock.advance(retry_after)
    assert limiter.check("id", capacity=n, refill_per_second=2.0)[0] is True
    assert limiter.check("id", capacity=n, refill_per_second=2.0)[0] is False


def test_partial_window_refill_is_insufficient_until_exact_boundary(clock):
    """Refilling for slightly less than the required time must still deny;
    only reaching (or passing) the exact boundary allows the next request."""
    limiter = RateLimiter("test")
    limiter.check("id", capacity=1, refill_per_second=1.0)  # exhaust the single token

    clock.advance(0.999)
    assert limiter.check("id", capacity=1, refill_per_second=1.0)[0] is False

    clock.advance(0.001)  # now exactly 1.0s elapsed since exhaustion
    assert limiter.check("id", capacity=1, refill_per_second=1.0)[0] is True


# ---------------------------------------------------------------------------
# Per-identity isolation under registry pressure (not just two identities).
# ---------------------------------------------------------------------------

def test_isolation_holds_across_many_simultaneous_identities(clock):
    """Exhausting many identities' budgets independently must not leak
    tokens between them even when all share one limiter instance."""
    limiter = RateLimiter("test", max_tracked=100)

    for i in range(20):
        identity = f"tenant:{i}"
        # Give each tenant a different capacity to make cross-contamination
        # detectable rather than coincidentally correct.
        capacity = (i % 5) + 1
        for _ in range(capacity):
            assert limiter.check(identity, capacity=capacity, refill_per_second=0.0001)[0] is True
        assert limiter.check(identity, capacity=capacity, refill_per_second=0.0001)[0] is False

    # Re-verify none of them regained tokens from another identity's checks.
    for i in range(20):
        identity = f"tenant:{i}"
        capacity = (i % 5) + 1
        assert limiter.check(identity, capacity=capacity, refill_per_second=0.0001)[0] is False


# ---------------------------------------------------------------------------
# Concurrency across the registry itself, not a single hot bucket.
# ---------------------------------------------------------------------------

def test_concurrent_distinct_identities_do_not_corrupt_the_registry(clock):
    """Many threads inserting many distinct identities concurrently must not
    corrupt OrderedDict state or lose/duplicate buckets - each identity's
    first request must succeed exactly once when capacity is 1."""
    limiter = RateLimiter("test", max_tracked=1000)

    def hit(i):
        identity = f"worker:{i}"
        first = limiter.check(identity, capacity=1, refill_per_second=0.0001)[0]
        second = limiter.check(identity, capacity=1, refill_per_second=0.0001)[0]
        return first, second

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(hit, range(200)))

    assert all(first is True and second is False for first, second in results)
    assert len(limiter) == 200
