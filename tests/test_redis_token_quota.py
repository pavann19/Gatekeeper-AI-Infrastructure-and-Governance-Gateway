"""
Unit and edge-case tests for the Redis-backed distributed token quota tracker (core/token_quota.py).
"""
import unittest.mock as mock

import pytest

from core import token_quota as tq


@pytest.fixture
def fake_redis():
    client = mock.MagicMock()
    mock_script = mock.MagicMock()
    client.register_script.return_value = mock_script
    return client


def test_redis_token_quota_would_exceed_unlimited(fake_redis):
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    assert tracker.would_exceed("acme", 0) is False
    fake_redis.get.assert_not_called()


def test_redis_token_quota_would_exceed_checks_redis(fake_redis):
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    fake_redis.get.return_value = b"99"

    assert tracker.would_exceed("acme", 100) is False

    fake_redis.get.return_value = b"100"
    assert tracker.would_exceed("acme", 100) is True


def test_redis_token_quota_usage_today(fake_redis):
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    fake_redis.get.return_value = b"350"

    assert tracker.usage_today("acme") == 350


def test_redis_token_quota_record_runs_script(fake_redis):
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    tracker.record("acme", 150)

    tracker._record_script.assert_called_once()
    args = tracker._record_script.call_args
    assert "acme" in args[1]["keys"][0]
    assert args[1]["args"][0] == "150"


def test_redis_token_quota_record_nonpositive_is_noop(fake_redis):
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    tracker.record("acme", 0)
    tracker.record("acme", -10)
    tracker._record_script.assert_not_called()


def test_redis_token_quota_record_eval_fallback():
    client = mock.MagicMock()
    client.register_script.side_effect = Exception("NOSCRIPT")
    tracker = tq.RedisTokenQuotaTracker(client)

    tracker.record("acme", 100)
    client.eval.assert_called_once()


def test_redis_token_quota_failover_to_local_on_error(fake_redis, caplog):
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    fake_redis.get.side_effect = ConnectionError("lost")
    tracker._record_script.side_effect = ConnectionError("lost")

    with caplog.at_level("ERROR"):
        # Record into local fallback
        tracker.record("acme", 50)
        assert tracker.would_exceed("acme", 100) is False
        tracker.record("acme", 60)
        assert tracker.would_exceed("acme", 100) is True
        assert tracker.usage_today("acme") == 110

    assert "falling back to local tracker" in caplog.text


def test_redis_token_quota_reset(fake_redis):
    fake_redis.scan_iter.return_value = ["gatekeeper:quota:a", "gatekeeper:quota:b"]
    tracker = tq.RedisTokenQuotaTracker(fake_redis)
    tracker.reset()

    fake_redis.scan_iter.assert_called_once_with("gatekeeper:quota:*")
    assert fake_redis.delete.call_count == 2


def test_build_token_quota_tracker_no_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    tracker = tq.build_token_quota_tracker()
    assert isinstance(tracker, tq.LocalTokenQuotaTracker)


def test_build_token_quota_tracker_reachable_redis(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.return_value = True

    with mock.patch("redis.from_url", return_value=fake_client):
        tracker = tq.build_token_quota_tracker()

    assert isinstance(tracker, tq.RedisTokenQuotaTracker)


def test_build_token_quota_tracker_unreachable_redis(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://unreachable:6379/0")

    with mock.patch("redis.from_url", side_effect=ConnectionError("refused")):
        with caplog.at_level("ERROR"):
            tracker = tq.build_token_quota_tracker()

    assert isinstance(tracker, tq.LocalTokenQuotaTracker)
    assert "falling back to local in-memory token quota tracker" in caplog.text
