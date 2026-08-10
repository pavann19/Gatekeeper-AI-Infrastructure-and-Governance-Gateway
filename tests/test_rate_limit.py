"""
Unit tests for the token-bucket limiter itself (core/rate_limit.py).

Endpoint-level enforcement is tested separately in tests/test_request_limits.py
— these tests are about the algorithm, and deliberately drive a controllable
clock rather than sleeping, so refill behaviour is asserted exactly instead of
approximately.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.rate_limit import RateLimiter, bucket_parameters


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


@pytest.fixture
def limiter():
    return RateLimiter("test")


# ---------------------------------------------------------------------------
# Core budget behaviour
# ---------------------------------------------------------------------------

def test_burst_is_allowed_up_to_capacity_then_refused(limiter, clock):
    """A caller may spend its whole burst at once, and not one request more."""
    for i in range(5):
        allowed, _ = limiter.check("alice", capacity=5, refill_per_second=1.0)
        assert allowed, f"request {i + 1} of the burst should have been allowed"

    allowed, retry_after = limiter.check("alice", capacity=5, refill_per_second=1.0)
    assert not allowed
    assert retry_after == pytest.approx(1.0), "one token/sec means a 1s wait"


def test_tokens_refill_at_the_configured_rate(limiter, clock):
    """Waiting restores budget proportionally, not all at once."""
    for _ in range(5):
        limiter.check("alice", capacity=5, refill_per_second=2.0)
    assert limiter.check("alice", capacity=5, refill_per_second=2.0)[0] is False

    clock.advance(0.5)  # 2/sec x 0.5s = exactly 1 token
    assert limiter.check("alice", capacity=5, refill_per_second=2.0)[0] is True
    # ...and that single token is now spent again.
    assert limiter.check("alice", capacity=5, refill_per_second=2.0)[0] is False


def test_refill_never_exceeds_capacity(limiter, clock):
    """An idle caller banks a burst, not an unbounded credit line."""
    limiter.check("alice", capacity=3, refill_per_second=1.0)
    clock.advance(10_000)  # far more than enough to overfill

    allowed = [limiter.check("alice", capacity=3, refill_per_second=1.0)[0] for _ in range(4)]
    assert allowed == [True, True, True, False], "capacity must cap the bank"


def test_retry_after_reflects_the_actual_wait(limiter, clock):
    """The value handed to Retry-After must be honest, or clients busy-loop."""
    for _ in range(2):
        limiter.check("bob", capacity=2, refill_per_second=0.5)

    _, retry_after = limiter.check("bob", capacity=2, refill_per_second=0.5)
    assert retry_after == pytest.approx(2.0), "0.5 tokens/sec means 2s for one token"

    # Waiting exactly that long must actually be sufficient.
    clock.advance(retry_after)
    assert limiter.check("bob", capacity=2, refill_per_second=0.5)[0] is True


def test_identities_have_independent_budgets(limiter, clock):
    """One caller exhausting its budget must not affect anyone else."""
    for _ in range(3):
        limiter.check("noisy", capacity=3, refill_per_second=1.0)
    assert limiter.check("noisy", capacity=3, refill_per_second=1.0)[0] is False

    assert limiter.check("quiet", capacity=3, refill_per_second=1.0)[0] is True


# ---------------------------------------------------------------------------
# Configuration edges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("capacity,refill", [(0, 1.0), (5, 0), (5, -1.0), (-1, 1.0)])
def test_nonpositive_configuration_disables_rather_than_blocks(limiter, capacity, refill):
    """
    A misconfigured limit must degrade to 'no limiting', never to 'total
    outage'. Settings validation rejects these values up front; this is the
    defence in depth behind it.
    """
    for _ in range(50):
        assert limiter.check("alice", capacity=capacity, refill_per_second=refill)[0] is True


def test_reset_clears_all_state(limiter, clock):
    for _ in range(3):
        limiter.check("alice", capacity=3, refill_per_second=1.0)
    assert limiter.check("alice", capacity=3, refill_per_second=1.0)[0] is False

    limiter.reset()

    assert len(limiter) == 0
    assert limiter.check("alice", capacity=3, refill_per_second=1.0)[0] is True


# ---------------------------------------------------------------------------
# Memory bound — the limiter must not become its own DoS vector
# ---------------------------------------------------------------------------

def test_registry_is_lru_capped(clock):
    """Rotating identities must not grow memory without bound."""
    limiter = RateLimiter("test", max_tracked=10)

    for i in range(500):
        limiter.check(f"ip:{i}", capacity=5, refill_per_second=1.0)

    assert len(limiter) == 10


def test_eviction_drops_the_least_recently_used(clock):
    """
    The victim must be the coldest caller, not an active one — otherwise a
    burst of new identities would reset the budget of whoever is currently
    being limited, which is precisely the abuse case.
    """
    limiter = RateLimiter("test", max_tracked=3)

    # 'active' spends its whole budget, then keeps touching its bucket while
    # other identities churn past it.
    for _ in range(2):
        limiter.check("active", capacity=2, refill_per_second=0.001)

    for i in range(10):
        limiter.check(f"churn:{i}", capacity=2, refill_per_second=0.001)
        limiter.check("active", capacity=2, refill_per_second=0.001)  # keep it warm

    # 'active' survived the churn, so it is still correctly out of budget.
    assert limiter.check("active", capacity=2, refill_per_second=0.001)[0] is False


# ---------------------------------------------------------------------------
# Concurrency — the property a lock exists to provide
# ---------------------------------------------------------------------------

def test_concurrent_callers_cannot_oversend(clock):
    """
    Twenty threads racing on one bucket of five tokens must yield exactly five
    successes. Without the lock, read-modify-write on `tokens` interleaves and
    lets extra requests through — the classic way a rate limiter silently
    fails to limit under exactly the load it was installed for.
    """
    limiter = RateLimiter("test")
    # Refill slow enough that no meaningful budget accrues during the test.
    args = dict(capacity=5, refill_per_second=0.0001)

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(
            lambda _: limiter.check("shared", **args)[0],
            range(20),
        ))

    assert sum(results) == 5


# ---------------------------------------------------------------------------
# rpm -> bucket conversion
# ---------------------------------------------------------------------------

def test_bucket_parameters_converts_rpm_to_per_second():
    capacity, refill = bucket_parameters(120.0, burst_seconds=10.0)
    assert refill == pytest.approx(2.0)
    assert capacity == pytest.approx(20.0)


def test_bucket_parameters_floors_capacity_at_one_token():
    """
    A limit so low that the burst rounds below one must still permit a single
    request. Capacity 0 would reject a caller who is within their sustained
    rate, which is not what any operator setting '2 per minute' intends.
    """
    capacity, refill = bucket_parameters(2.0, burst_seconds=1.0)
    assert capacity == 1.0
    assert refill == pytest.approx(2.0 / 60.0)
