"""
Unit and edge-case tests for the Redis-backed distributed rate limiter (core/rate_limit.py).

Verifies:
  - Atomic Lua script invocation and parameter formatting.
  - Return value decoding (allowed status and retry_after float).
  - Non-positive capacity and refill-rate bypasses.
  - Graceful fallback to in-process LocalRateLimiter on Redis connection/eval failures.
  - Scoped reset and scan-based length counting.
  - Dynamic backend selection in build_rate_limiter (REDIS_URL presence, ping checks, credential redaction).
"""
import unittest.mock as mock

import pytest

from core import rate_limit as rl


@pytest.fixture
def fake_redis():
    client = mock.MagicMock()
    # Mock register_script to return a mock script object
    mock_script = mock.MagicMock()
    client.register_script.return_value = mock_script
    return client


# ---------------------------------------------------------------------------
# RedisRateLimiter core check behavior
# ---------------------------------------------------------------------------

def test_redis_rate_limiter_check_allowed_returns_true_zero(fake_redis):
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")
    limiter._script.return_value = [1, b"0.0"]

    allowed, retry_after = limiter.check("alice", capacity=5.0, refill_per_second=1.0, now=100.0)

    assert allowed is True
    assert retry_after == 0.0
    limiter._script.assert_called_once_with(
        keys=["gatekeeper:ratelimit:assess:alice"],
        args=["5.0", "1.0", "100.0"],
    )


def test_redis_rate_limiter_check_denied_returns_false_and_retry_after(fake_redis):
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")
    limiter._script.return_value = [0, b"1.75"]

    allowed, retry_after = limiter.check("bob", capacity=2.0, refill_per_second=0.5, now=200.0)

    assert allowed is False
    assert retry_after == pytest.approx(1.75)
    limiter._script.assert_called_once_with(
        keys=["gatekeeper:ratelimit:assess:bob"],
        args=["2.0", "0.5", "200.0"],
    )


def test_redis_rate_limiter_string_retry_after_decoded_correctly(fake_redis):
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")
    limiter._script.return_value = [0, "0.500"]

    allowed, retry_after = limiter.check("charlie", capacity=1.0, refill_per_second=2.0)

    assert allowed is False
    assert retry_after == pytest.approx(0.5)


def test_redis_rate_limiter_empty_now_passes_empty_string(fake_redis):
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")
    limiter._script.return_value = [1, b"0.0"]

    limiter.check("dave", capacity=10.0, refill_per_second=1.0, now=None)

    # When now is None, empty string is passed so Lua script uses redis.call('TIME')
    limiter._script.assert_called_once_with(
        keys=["gatekeeper:ratelimit:assess:dave"],
        args=["10.0", "1.0", ""],
    )


@pytest.mark.parametrize("capacity,refill", [(0, 1.0), (5, 0), (5, -1.0), (-1, 1.0)])
def test_redis_rate_limiter_nonpositive_config_bypasses_redis(fake_redis, capacity, refill):
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")

    allowed, retry_after = limiter.check("alice", capacity=capacity, refill_per_second=refill)

    assert allowed is True
    assert retry_after == 0.0
    fake_redis.register_script.return_value.assert_not_called()
    fake_redis.eval.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback to eval when script registration fails
# ---------------------------------------------------------------------------

def test_redis_rate_limiter_registration_failure_falls_back_to_eval():
    client = mock.MagicMock()
    client.register_script.side_effect = Exception("NOSCRIPT")
    client.eval.return_value = [1, b"0.0"]

    limiter = rl.RedisRateLimiter(client, name="assess")
    assert limiter._script is None

    allowed, retry_after = limiter.check("alice", capacity=5.0, refill_per_second=1.0, now=100.0)

    assert allowed is True
    assert retry_after == 0.0
    client.eval.assert_called_once_with(
        rl.LUA_TOKEN_BUCKET,
        1,
        "gatekeeper:ratelimit:assess:alice",
        "5.0",
        "1.0",
        "100.0",
    )


# ---------------------------------------------------------------------------
# Resilience & Runtime Error Fallback
# ---------------------------------------------------------------------------

def test_redis_rate_limiter_runtime_error_falls_back_to_local_limiter(fake_redis, caplog):
    """When Redis has a runtime glitch or disconnects, it falls back to the in-memory limiter."""
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")
    limiter._script.side_effect = ConnectionError("Redis connection lost")

    with caplog.at_level("ERROR"):
        # First call on local fallback spends 1 token from capacity 2 -> allowed
        allowed1, retry1 = limiter.check("tenant-a", capacity=2.0, refill_per_second=0.0001, now=10.0)
        # Second call on local fallback spends 2nd token -> allowed
        allowed2, retry2 = limiter.check("tenant-a", capacity=2.0, refill_per_second=0.0001, now=10.0)
        # Third call on local fallback exceeds capacity 2 -> denied
        allowed3, retry3 = limiter.check("tenant-a", capacity=2.0, refill_per_second=0.0001, now=10.0)

    assert allowed1 is True and retry1 == 0.0
    assert allowed2 is True and retry2 == 0.0
    assert allowed3 is False and retry3 > 0.0
    assert "Redis rate limiter check failed" in caplog.text


# ---------------------------------------------------------------------------
# Reset & Length
# ---------------------------------------------------------------------------

def test_redis_rate_limiter_reset_deletes_only_scoped_keys(fake_redis):
    fake_redis.scan_iter.return_value = [
        "gatekeeper:ratelimit:assess:alice",
        "gatekeeper:ratelimit:assess:bob",
    ]
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")

    limiter.reset()

    fake_redis.scan_iter.assert_called_once_with("gatekeeper:ratelimit:assess:*")
    assert fake_redis.delete.call_count == 2
    # Must never call blanket flushdb
    assert not hasattr(fake_redis, "flushdb") or not fake_redis.flushdb.called


def test_redis_rate_limiter_reset_error_logs_and_continues(fake_redis, caplog):
    fake_redis.scan_iter.side_effect = TimeoutError("scan timeout")
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")

    with caplog.at_level("ERROR"):
        limiter.reset()  # Must not raise

    assert "Redis rate limiter reset failed" in caplog.text


def test_redis_rate_limiter_len_counts_scanned_keys(fake_redis):
    fake_redis.scan_iter.return_value = [
        "gatekeeper:ratelimit:assess:k1",
        "gatekeeper:ratelimit:assess:k2",
        "gatekeeper:ratelimit:assess:k3",
    ]
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")

    assert len(limiter) == 3
    fake_redis.scan_iter.assert_called_once_with("gatekeeper:ratelimit:assess:*")


def test_redis_rate_limiter_len_error_falls_back_to_local_len(fake_redis, caplog):
    fake_redis.scan_iter.side_effect = ConnectionError("disconnected")
    limiter = rl.RedisRateLimiter(fake_redis, name="assess")

    with caplog.at_level("ERROR"):
        assert len(limiter) == 0

    assert "Redis rate limiter __len__ failed" in caplog.text


# ---------------------------------------------------------------------------
# build_rate_limiter factory selection and fallback
# ---------------------------------------------------------------------------

def test_build_rate_limiter_no_redis_url_selects_local(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    limiter = rl.build_rate_limiter("assess")
    assert isinstance(limiter, rl.LocalRateLimiter)
    assert limiter.name == "assess"


def test_build_rate_limiter_reachable_redis_selects_redis(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.return_value = True

    with mock.patch("redis.from_url", return_value=fake_client):
        limiter = rl.build_rate_limiter("assess")

    assert isinstance(limiter, rl.RedisRateLimiter)
    assert limiter.name == "assess"


def test_build_rate_limiter_unreachable_redis_falls_back_to_local(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://unreachable:6379/0")

    with mock.patch("redis.from_url", side_effect=ConnectionError("refused")):
        with caplog.at_level("ERROR"):
            limiter = rl.build_rate_limiter("assess")

    assert isinstance(limiter, rl.LocalRateLimiter)
    assert "falling back to local in-memory rate limiter" in caplog.text


def test_build_rate_limiter_ping_timeout_falls_back_to_local(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.side_effect = TimeoutError("ping timeout")

    with mock.patch("redis.from_url", return_value=fake_client):
        with caplog.at_level("ERROR"):
            limiter = rl.build_rate_limiter("assess")

    assert isinstance(limiter, rl.LocalRateLimiter)
    assert "falling back to local in-memory rate limiter" in caplog.text


def test_build_rate_limiter_missing_redis_package_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("No module named 'redis'")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        limiter = rl.build_rate_limiter("assess")

    assert isinstance(limiter, rl.LocalRateLimiter)


def test_build_rate_limiter_startup_log_redacts_credentials(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://operator:verysecretpass@10.0.0.1:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.return_value = True

    with mock.patch("redis.from_url", return_value=fake_client):
        with caplog.at_level("INFO"):
            rl.build_rate_limiter("assess")

    assert "verysecretpass" not in caplog.text
    assert "10.0.0.1:6379/0" in caplog.text


def test_build_rate_limiter_error_log_does_not_leak_credentials(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://operator:verysecretpass@unreachable:6379/0")

    with mock.patch("redis.from_url", side_effect=ConnectionError("connect failed")):
        with caplog.at_level("ERROR"):
            rl.build_rate_limiter("assess")

    assert "verysecretpass" not in caplog.text


# ---------------------------------------------------------------------------
# Stateful Token Bucket Simulation
# ---------------------------------------------------------------------------

class StatefulRedisMock:
    """Emulates Redis hash storage and executes Python equivalent of the Lua token bucket."""

    def __init__(self):
        self.hashes = {}
        self.ttls = {}

    def register_script(self, script_str):
        def _execute(keys, args):
            key = keys[0]
            capacity = float(args[0])
            refill_per_second = float(args[1])
            now_str = args[2]
            now = float(now_str) if now_str else 1000.0

            data = self.hashes.get(key)
            if data is None:
                tokens = capacity
                last_refill = now
            else:
                tokens = float(data["tokens"])
                last_refill = float(data["last_refill"])
                elapsed = now - last_refill
                if elapsed > 0:
                    tokens = min(capacity, tokens + (elapsed * refill_per_second))
                    last_refill = now

            if tokens >= 1.0:
                tokens -= 1.0
                allowed = 1
                retry_after = 0.0
            else:
                allowed = 0
                deficit = 1.0 - tokens
                retry_after = deficit / refill_per_second

            self.hashes[key] = {"tokens": str(tokens), "last_refill": str(last_refill)}
            fill_time = int(capacity / refill_per_second) + 1
            self.ttls[key] = max(3600, fill_time * 2)

            return [allowed, f"{retry_after:.6f}".encode("utf-8")]

        return _execute

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        for k in list(self.hashes.keys()):
            if k.startswith(prefix):
                yield k

    def delete(self, key):
        self.hashes.pop(key, None)
        self.ttls.pop(key, None)


def test_redis_rate_limiter_stateful_simulation():
    stateful = StatefulRedisMock()
    limiter = rl.RedisRateLimiter(stateful, name="assess")

    # 1. Burst of 3 tokens from capacity 3
    t = 1000.0
    for i in range(3):
        allowed, retry = limiter.check("user1", capacity=3.0, refill_per_second=1.0, now=t)
        assert allowed is True, f"Request {i+1} should be allowed"
        assert retry == 0.0

    # 2. 4th request in same instant is denied
    allowed, retry = limiter.check("user1", capacity=3.0, refill_per_second=1.0, now=t)
    assert allowed is False
    assert retry == pytest.approx(1.0)

    # 3. Advance clock by 0.5s (0.5 token refilled, still < 1 token)
    t += 0.5
    allowed, retry = limiter.check("user1", capacity=3.0, refill_per_second=1.0, now=t)
    assert allowed is False
    assert retry == pytest.approx(0.5)

    # 4. Advance clock by another 0.5s (total 1.0 token refilled) -> allowed!
    t += 0.5
    allowed, retry = limiter.check("user1", capacity=3.0, refill_per_second=1.0, now=t)
    assert allowed is True
    assert retry == 0.0

    # 5. User 2 has independent bucket
    allowed2, _ = limiter.check("user2", capacity=3.0, refill_per_second=1.0, now=t)
    assert allowed2 is True
    assert len(limiter) == 2

    # 6. Reset clears state
    limiter.reset()
    assert len(limiter) == 0
