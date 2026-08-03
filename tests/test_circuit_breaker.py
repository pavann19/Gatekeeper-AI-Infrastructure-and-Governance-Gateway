"""
Tests for core/circuit_breaker.py.

The property that matters most: after the backend recovers, the breaker must
let traffic through again on its own (half-open probe) — a breaker that never
recovers without a manual reset is worse than no breaker at all.
"""
import time

from core.circuit_breaker import CircuitBreaker


def make_breaker(threshold=3, cooldown=0.05):
    return CircuitBreaker("test", failure_threshold=threshold, cooldown_seconds=cooldown)


def test_starts_closed():
    b = make_breaker()
    assert b.is_open() is False


def test_stays_closed_below_threshold():
    b = make_breaker(threshold=3)
    b.record_failure()
    b.record_failure()
    assert b.is_open() is False  # only 2 of 3


def test_opens_at_threshold():
    b = make_breaker(threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_failure()
    assert b.is_open() is True


def test_success_resets_the_failure_count():
    b = make_breaker(threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    b.record_failure()
    assert b.is_open() is False  # count reset to 0 by the success, only 2 since


def test_half_open_probe_after_cooldown():
    """THE PROPERTY THAT MATTERS MOST: the breaker must self-heal."""
    b = make_breaker(threshold=1, cooldown=0.05)
    b.record_failure()
    assert b.is_open() is True

    time.sleep(0.06)
    assert b.is_open() is False  # cooldown elapsed -> probe allowed through


def test_open_before_cooldown_elapses():
    b = make_breaker(threshold=1, cooldown=10)
    b.record_failure()
    assert b.is_open() is True
    assert b.is_open() is True  # still well within the 10s cooldown


def test_failed_probe_does_not_instantly_reopen():
    """
    Documents the deliberate design: one flaky failure right after a
    half-open probe does not immediately force another full cooldown — it
    takes `failure_threshold` consecutive failures again, same as the
    original trip. See CircuitBreaker.record_failure's docstring for why.
    """
    b = make_breaker(threshold=2, cooldown=0.05)
    b.record_failure()
    b.record_failure()
    assert b.is_open() is True

    time.sleep(0.06)
    assert b.is_open() is False  # half-open probe window
    b.record_failure()  # the probe itself fails
    assert b.is_open() is False  # only 1 failure since the reset; not re-tripped yet
    b.record_failure()
    assert b.is_open() is True  # now at threshold again


def test_successful_probe_fully_recovers():
    b = make_breaker(threshold=1, cooldown=0.05)
    b.record_failure()
    time.sleep(0.06)
    assert b.is_open() is False
    b.record_success()
    assert b.is_open() is False
    b.record_failure()
    assert b.is_open() is True  # back to needing a fresh trip, not a partial one


def test_reset_forces_closed_regardless_of_state():
    b = make_breaker(threshold=1, cooldown=1000)
    b.record_failure()
    assert b.is_open() is True
    b.reset()
    assert b.is_open() is False


def test_breakers_are_independent_per_instance():
    a = make_breaker(threshold=1)
    b = make_breaker(threshold=1)
    a.record_failure()
    assert a.is_open() is True
    assert b.is_open() is False
