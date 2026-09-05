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


# --- shared (Redis-backed) breaker state --------------------------------

class _FakeRedis:
    """Minimal in-memory stand-in that actually executes the two Lua scripts'
    semantics, so a shared store can be handed to two breaker instances to
    simulate two replicas."""

    def __init__(self, store=None, clock=None):
        self.store = store if store is not None else {}
        self.clock = clock if clock is not None else [1000.0]

    # redis-py surface used by CircuitBreaker
    def get(self, k):
        v = self.store.get(k)
        return None if v is None else str(v).encode()

    def set(self, k, v):
        self.store[k] = str(v)

    def delete(self, *ks):
        for k in ks:
            self.store.pop(k, None)

    def register_script(self, src):
        return self._record_failure if "INCR" in src else self._open_check

    def _now(self):
        return self.clock[0]

    def _open_check(self, keys, args):
        opened = self.store.get(keys[0])
        if opened is None:
            return 0
        if self._now() - float(opened) >= float(args[0]):
            self.store.pop(keys[0], None)
            self.store.pop(keys[1], None)
            return 0
        return 1

    def _record_failure(self, keys, args):
        fails = int(self.store.get(keys[1], 0)) + 1
        self.store[keys[1]] = fails
        if self.store.get(keys[0]) is None and fails >= int(args[0]):
            self.store[keys[0]] = self._now()
            return [fails, 1]
        return [fails, 0]


def _shared_pair(threshold=3, cooldown=30, clock=None):
    store, clock = {}, (clock if clock is not None else [1000.0])
    r1, r2 = _FakeRedis(store, clock), _FakeRedis(store, clock)
    a = CircuitBreaker("judge", failure_threshold=threshold,
                       cooldown_seconds=cooldown, redis_client=r1)
    b = CircuitBreaker("judge", failure_threshold=threshold,
                       cooldown_seconds=cooldown, redis_client=r2)
    return a, b, clock


def test_shared_one_replica_trip_is_seen_by_the_other():
    a, b, _ = _shared_pair(threshold=3)
    a.record_failure()
    a.record_failure()
    a.record_failure()
    assert a.is_open() is True
    assert b.is_open() is True  # replica B never saw a failure locally


def test_shared_failures_accumulate_across_replicas():
    a, b, _ = _shared_pair(threshold=3)
    a.record_failure()
    b.record_failure()
    assert b.is_open() is False  # 2 of 3
    a.record_failure()           # third failure, on replica A
    assert b.is_open() is True


def test_shared_half_open_then_failed_probe_reopens_for_all():
    clock = [1000.0]
    a, b, _ = _shared_pair(threshold=1, cooldown=30, clock=clock)
    a.record_failure()
    assert a.is_open() is True and b.is_open() is True

    clock[0] += 31  # cooldown elapsed -> first caller's is_open() clears state
    assert a.is_open() is False  # probe window: a real call is allowed through

    b.record_failure()  # that probe call fails, on whichever replica ran it
    assert a.is_open() is True  # re-opened, and every replica sees it
    assert b.is_open() is True


def test_shared_success_on_one_replica_clears_for_all():
    a, b, _ = _shared_pair(threshold=2)
    a.record_failure()
    b.record_failure()
    assert a.is_open() is True
    b.record_success()
    assert a.is_open() is False
    assert a._consecutive_failures == 0


def test_shared_reads_have_no_half_open_side_effect():
    clock = [1000.0]
    a, b, _ = _shared_pair(threshold=1, cooldown=30, clock=clock)
    a.record_failure()
    clock[0] += 31
    # metrics / health read _opened_at directly; it must not consume the probe
    assert a._opened_at is not None
    assert a._opened_at is not None
    assert a.is_open() is False  # the probe is still available for a real call


def test_falls_back_to_local_when_redis_raises():
    class _BrokenRedis:
        def register_script(self, src):
            def _boom(*a, **k):
                raise ConnectionError("redis down")
            return _boom
        def get(self, *a, **k):
            raise ConnectionError("redis down")
        def delete(self, *a, **k):
            raise ConnectionError("redis down")

    b = CircuitBreaker("judge", failure_threshold=2, cooldown_seconds=10,
                       redis_client=_BrokenRedis())
    # every Redis call raises -> behaviour is exactly the process-local breaker
    b.record_failure()
    b.record_failure()
    assert b.is_open() is True
    b.reset()
    assert b.is_open() is False
