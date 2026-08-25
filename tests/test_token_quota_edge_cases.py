"""Additional edge-case coverage for core/token_quota.py, complementing
tests/test_token_quota.py: exact boundary checks, UTC-midnight reset
behavior, seconds_until_utc_midnight correctness, per-tenant isolation,
record() accumulation, and the quota<=0 "unlimited" sentinel."""
from datetime import datetime, timedelta, timezone

import core.token_quota as token_quota
from core.token_quota import TokenQuotaTracker, seconds_until_utc_midnight


class _FixedDatetime(datetime):
    """A datetime subclass whose now() is pinned, so module code calling
    `datetime.now(timezone.utc)` sees a controlled clock without touching
    the real system time."""
    _fixed = None

    @classmethod
    def now(cls, tz=None):
        return cls._fixed if tz is None else cls._fixed.astimezone(tz)


def _patch_clock(monkeypatch, fixed_utc: datetime):
    frozen = _FixedDatetime
    frozen._fixed = fixed_utc
    monkeypatch.setattr(token_quota, "datetime", frozen)


# --- would_exceed: exact boundary behavior ---

def test_would_exceed_false_one_token_under_quota():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 99)
    assert tracker.would_exceed("acme", 100) is False


def test_would_exceed_true_at_exactly_quota():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 100)
    assert tracker.would_exceed("acme", 100) is True


def test_would_exceed_true_one_token_over_quota():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 101)
    assert tracker.would_exceed("acme", 100) is True


def test_would_exceed_false_with_zero_usage_against_positive_quota():
    tracker = TokenQuotaTracker()
    assert tracker.would_exceed("acme", 1) is False


# --- quota<=0 "unlimited" sentinel: always False regardless of usage ---

def test_unlimited_sentinel_zero_quota_always_false_even_at_huge_usage():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 999_999)
    assert tracker.would_exceed("acme", 0) is False


def test_unlimited_sentinel_negative_quota_always_false():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 50)
    assert tracker.would_exceed("acme", -1) is False


def test_unlimited_sentinel_zero_quota_false_with_no_usage():
    tracker = TokenQuotaTracker()
    assert tracker.would_exceed("acme", 0) is False


# --- record() accumulation correctness across multiple calls ---

def test_record_accumulates_across_many_calls():
    tracker = TokenQuotaTracker()
    for _ in range(10):
        tracker.record("acme", 7)
    assert tracker.usage_today("acme") == 70


def test_record_accumulation_reflected_in_would_exceed():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 40)
    tracker.record("acme", 40)
    assert tracker.would_exceed("acme", 100) is False
    tracker.record("acme", 20)
    assert tracker.usage_today("acme") == 100
    assert tracker.would_exceed("acme", 100) is True


def test_record_mixes_positive_calls_with_no_op_negative_and_zero():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 30)
    tracker.record("acme", -100)
    tracker.record("acme", 0)
    tracker.record("acme", 20)
    assert tracker.usage_today("acme") == 50


# --- per-tenant isolation ---

def test_two_tenants_usage_and_quota_checks_never_cross_contaminate():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 90)
    tracker.record("globex", 10)

    assert tracker.usage_today("acme") == 90
    assert tracker.usage_today("globex") == 10
    assert tracker.would_exceed("acme", 100) is False
    assert tracker.would_exceed("globex", 100) is False

    tracker.record("acme", 15)
    # globex must be unaffected by acme's subsequent recording
    assert tracker.usage_today("globex") == 10
    assert tracker.would_exceed("acme", 100) is True
    assert tracker.would_exceed("globex", 100) is False


def test_reset_of_shared_tracker_clears_every_tenant_not_just_one():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 100)
    tracker.record("globex", 200)
    tracker.reset()
    assert tracker.usage_today("acme") == 0
    assert tracker.usage_today("globex") == 0


# --- UTC-midnight reset behavior ---

def test_usage_from_yesterday_does_not_count_against_todays_quota(monkeypatch):
    tracker = TokenQuotaTracker()

    day1 = datetime(2026, 8, 24, 23, 59, 0, tzinfo=timezone.utc)
    _patch_clock(monkeypatch, day1)
    tracker.record("acme", 90)
    assert tracker.usage_today("acme") == 90
    assert tracker.would_exceed("acme", 100) is False

    # Cross the UTC midnight boundary into the next day.
    day2 = datetime(2026, 8, 25, 0, 0, 5, tzinfo=timezone.utc)
    _patch_clock(monkeypatch, day2)
    # Yesterday's 90 must not carry over.
    assert tracker.usage_today("acme") == 0
    assert tracker.would_exceed("acme", 100) is False

    tracker.record("acme", 50)
    assert tracker.usage_today("acme") == 50


def test_internal_day_rollover_replaces_tenant_usage_object(monkeypatch):
    tracker = TokenQuotaTracker()

    day1 = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    _patch_clock(monkeypatch, day1)
    tracker.record("acme", 500)
    stale_usage = tracker._usage["acme"]
    assert stale_usage.day == "2026-08-24"
    assert stale_usage.tokens_used == 500

    day2 = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)
    _patch_clock(monkeypatch, day2)
    tracker.record("acme", 5)
    fresh_usage = tracker._usage["acme"]
    assert fresh_usage.day == "2026-08-25"
    assert fresh_usage.tokens_used == 5
    # The stale object itself must be untouched by the rollover.
    assert stale_usage.tokens_used == 500


def test_one_tenant_rolling_over_does_not_affect_another_still_on_prior_day(monkeypatch):
    tracker = TokenQuotaTracker()

    day1 = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
    _patch_clock(monkeypatch, day1)
    tracker.record("acme", 40)
    tracker.record("globex", 60)

    day2 = datetime(2026, 8, 25, 10, 0, 0, tzinfo=timezone.utc)
    _patch_clock(monkeypatch, day2)
    # Only touch acme; globex's stored record is still keyed to "yesterday".
    tracker.record("acme", 1)
    assert tracker.usage_today("acme") == 1
    # globex, once queried "today", also rolls over to zero (new day, no
    # usage recorded yet) rather than reporting the old day's total.
    assert tracker.usage_today("globex") == 0


# --- seconds_until_utc_midnight correctness at several points in a day ---

def test_seconds_until_midnight_at_start_of_day(monkeypatch):
    _patch_clock(monkeypatch, datetime(2026, 8, 25, 0, 0, 0, tzinfo=timezone.utc))
    assert seconds_until_utc_midnight() == 86400.0


def test_seconds_until_midnight_at_noon(monkeypatch):
    _patch_clock(monkeypatch, datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc))
    assert seconds_until_utc_midnight() == 43200.0


def test_seconds_until_midnight_one_second_before(monkeypatch):
    _patch_clock(monkeypatch, datetime(2026, 8, 25, 23, 59, 59, tzinfo=timezone.utc))
    assert seconds_until_utc_midnight() == 1.0


def test_seconds_until_midnight_with_microseconds_rounds_down_correctly(monkeypatch):
    _patch_clock(monkeypatch, datetime(2026, 8, 25, 23, 59, 59, 500000, tzinfo=timezone.utc))
    assert seconds_until_utc_midnight() == 0.5


def test_seconds_until_midnight_never_negative_at_exact_midnight(monkeypatch):
    _patch_clock(monkeypatch, datetime(2026, 8, 25, 0, 0, 0, 0, tzinfo=timezone.utc))
    result = seconds_until_utc_midnight()
    assert result >= 0.0
    assert result == 86400.0


def test_seconds_until_midnight_early_morning(monkeypatch):
    _patch_clock(monkeypatch, datetime(2026, 8, 25, 1, 30, 0, tzinfo=timezone.utc))
    expected = timedelta(hours=22, minutes=30).total_seconds()
    assert seconds_until_utc_midnight() == expected
