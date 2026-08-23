"""Unit tests for core/token_quota.py (Phase 5: token accounting)."""
from core.token_quota import TokenQuotaTracker, extract_total_tokens


def test_unlimited_quota_never_exceeds():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 10_000_000)
    assert tracker.would_exceed("acme", 0) is False


def test_records_and_checks_against_quota():
    tracker = TokenQuotaTracker()
    assert tracker.would_exceed("acme", 100) is False
    tracker.record("acme", 60)
    assert tracker.would_exceed("acme", 100) is False
    tracker.record("acme", 60)
    assert tracker.would_exceed("acme", 100) is True


def test_usage_today_reports_running_total():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 30)
    tracker.record("acme", 20)
    assert tracker.usage_today("acme") == 50


def test_tenants_are_tracked_independently():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 100)
    assert tracker.usage_today("acme") == 100
    assert tracker.usage_today("other") == 0
    assert tracker.would_exceed("other", 50) is False


def test_non_positive_record_is_a_no_op():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 0)
    tracker.record("acme", -5)
    assert tracker.usage_today("acme") == 0


def test_reset_clears_all_tenants():
    tracker = TokenQuotaTracker()
    tracker.record("acme", 100)
    tracker.reset()
    assert tracker.usage_today("acme") == 0


def test_lru_eviction_bounds_memory():
    tracker = TokenQuotaTracker(max_tracked=2)
    tracker.record("a", 1)
    tracker.record("b", 1)
    tracker.record("c", 1)
    assert len(tracker._usage) == 2
    assert "a" not in tracker._usage


# --- extract_total_tokens: normalising provider-specific usage shapes ---

def test_extract_total_tokens_openai_shape():
    assert extract_total_tokens({"total_tokens": 42, "prompt_tokens": 10}) == 42


def test_extract_total_tokens_anthropic_shape():
    assert extract_total_tokens({"input_tokens": 10, "output_tokens": 15}) == 25


def test_extract_total_tokens_anthropic_shape_partial():
    assert extract_total_tokens({"input_tokens": 10}) == 10


def test_extract_total_tokens_none_usage():
    assert extract_total_tokens(None) == 0


def test_extract_total_tokens_unrecognised_shape_degrades_to_zero():
    assert extract_total_tokens({"weird_field": 99}) == 0


def test_extract_total_tokens_non_dict_degrades_to_zero():
    assert extract_total_tokens("not a dict") == 0
