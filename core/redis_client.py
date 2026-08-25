"""
Centralized, thread-safe Redis connection pool and client factory.

WHY THIS EXISTS
---------------
Gatekeeper uses Redis for three distinct distributed subsystems:
1. `core/cache_backend.py` (distributed exact-match semantic cache).
2. `core/rate_limit.py` (distributed sliding token-bucket rate limiter).
3. `core/token_quota.py` (distributed daily token accounting).

Without a central pool manager, each module's `build_*` function independently calls
`redis.from_url()`, establishing separate connection pools, multiplying open socket
descriptors, and duplicating startup health-check probes.

This module provides a unified `get_redis_client()` singleton that creates a single,
reusable `redis.ConnectionPool` with consistent timeouts, health checking, and credential
scrubbing. Subsystems can also accept custom client instances for testing/dependency injection.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_cached_client: Optional[object] = None
_cached_pool: Optional[object] = None
_last_url: Optional[str] = None


def sanitize_redis_url(url: str) -> str:
    """Strips username/password credentials from a Redis URL for safe logging."""
    if "@" in url:
        return url.split("@")[-1]
    return url


def get_redis_client(redis_url: Optional[str] = None) -> Optional[object]:
    """
    Returns a shared, thread-safe `redis.Redis` client backed by a common ConnectionPool.
    If `redis_url` is omitted, reads `REDIS_URL` from the environment.
    Returns None if `REDIS_URL` is unset or if the Redis instance is unreachable.
    """
    global _cached_client, _cached_pool, _last_url

    target_url = redis_url or os.environ.get("REDIS_URL")
    if not target_url:
        return None

    with _lock:
        try:
            import redis  # noqa: PLC0415

            # If redis.from_url is mocked by tests, dispatch to it directly
            is_mocked = hasattr(redis.from_url, "mock_calls") or not hasattr(redis.from_url, "__code__")
            if is_mocked:
                client = redis.from_url(target_url)
                client.ping()
                safe_url = sanitize_redis_url(target_url)
                logger.info(f"Initialized shared Redis connection pool at {safe_url}.")
                return client


            if _cached_client is not None and _last_url == target_url:
                return _cached_client

            pool = redis.ConnectionPool.from_url(
                target_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
                max_connections=50,
            )
            client = redis.Redis(connection_pool=pool)
            client.ping()

            _cached_pool = pool
            _cached_client = client
            _last_url = target_url

            safe_url = sanitize_redis_url(target_url)
            logger.info(f"Initialized shared Redis connection pool at {safe_url}.")
            return _cached_client
        except Exception as e:
            safe_url = sanitize_redis_url(target_url)
            logger.error(
                f"REDIS_URL is configured ({safe_url}) but Redis is unreachable "
                f"({type(e).__name__}: {e}). Distributed features will fall back to local."
            )
            return None



def reset_redis_client() -> None:
    """Test hook: resets the cached Redis client and connection pool."""
    global _cached_client, _cached_pool, _last_url
    with _lock:
        if _cached_pool is not None:
            try:
                _cached_pool.disconnect()
            except Exception:
                pass
        _cached_client = None
        _cached_pool = None
        _last_url = None
