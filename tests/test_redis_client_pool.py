"""
Unit tests for the centralized Redis connection pool and factory in `core/redis_client.py`.
Verifies singleton reuse, graceful fallback when unreachable/unset, and shared
connection pool inheritance across cache, rate limiting, and token quota subsystems.
"""
import os
from unittest.mock import MagicMock, patch

from core.cache_backend import RedisExactCache, build_exact_cache_backend
from core.rate_limit import RedisRateLimiter, build_rate_limiter
from core.redis_client import (
    get_redis_client,
    reset_redis_client,
    sanitize_redis_url,
)
from core.token_quota import RedisTokenQuotaTracker, build_token_quota_tracker


def setup_function():
    reset_redis_client()


def teardown_function():
    reset_redis_client()


def test_sanitize_redis_url():
    assert sanitize_redis_url("redis://localhost:6379/0") == "redis://localhost:6379/0"
    assert sanitize_redis_url("redis://:secret_password@redis-host:6379/1") == "redis-host:6379/1"
    assert sanitize_redis_url("redis://user:p4ssw0rd@10.0.0.5:6380/2") == "10.0.0.5:6380/2"


def test_get_redis_client_returns_none_when_unset():
    with patch.dict(os.environ, {}, clear=True):
        assert get_redis_client() is None


def test_get_redis_client_returns_none_on_connection_error():
    with patch.dict(os.environ, {"REDIS_URL": "redis://unreachable-host:6379/0"}):
        client = get_redis_client()
        assert client is None


def test_get_redis_client_reuses_pool_and_singleton():
    fake_pool = MagicMock()
    fake_client = MagicMock()
    fake_client.connection_pool = fake_pool
    fake_client.ping.return_value = True

    with patch("redis.ConnectionPool.from_url", return_value=fake_pool) as mock_pool_factory, \
         patch("redis.Redis", return_value=fake_client) as mock_redis_cls:

        client1 = get_redis_client("redis://mock-redis:6379/0")
        client2 = get_redis_client("redis://mock-redis:6379/0")

        assert client1 is client2
        assert client1 is fake_client
        mock_pool_factory.assert_called_once()
        mock_redis_cls.assert_called_once_with(connection_pool=fake_pool)
        fake_client.ping.assert_called_once()


def test_subsystems_share_identical_redis_client():
    fake_pool = MagicMock()
    fake_client = MagicMock()
    fake_client.connection_pool = fake_pool
    fake_client.ping.return_value = True

    with patch.dict(os.environ, {"REDIS_URL": "redis://mock-redis:6379/0"}), \
         patch("redis.ConnectionPool.from_url", return_value=fake_pool), \
         patch("redis.Redis", return_value=fake_client):

        cache_backend = build_exact_cache_backend()
        rate_limiter = build_rate_limiter("assess")
        mcp_limiter = build_rate_limiter("mcp")
        quota_tracker = build_token_quota_tracker()

        assert isinstance(cache_backend, RedisExactCache)
        assert isinstance(rate_limiter, RedisRateLimiter)
        assert isinstance(mcp_limiter, RedisRateLimiter)
        assert isinstance(quota_tracker, RedisTokenQuotaTracker)

        # Confirm all 4 subsystems point to the SAME shared client and connection pool
        assert cache_backend._client is fake_client
        assert rate_limiter._client is fake_client
        assert mcp_limiter._client is fake_client
        assert quota_tracker._client is fake_client


def test_reset_redis_client_disconnects_pool():
    fake_pool = MagicMock()
    fake_client = MagicMock()
    fake_client.connection_pool = fake_pool
    fake_client.ping.return_value = True

    with patch.dict(os.environ, {"REDIS_URL": "redis://mock-redis:6379/0"}), \
         patch("redis.ConnectionPool.from_url", return_value=fake_pool), \
         patch("redis.Redis", return_value=fake_client):

        get_redis_client()
        reset_redis_client()

        fake_pool.disconnect.assert_called_once()
        # Next call creates a new client
        assert get_redis_client() is fake_client
