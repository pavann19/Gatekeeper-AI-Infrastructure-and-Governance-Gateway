"""
Tests for the pluggable exact-match cache backend (core/cache_backend.py).

The property that matters most: backend SELECTION must never crash the
process and must always yield a working cache — Redis configured-but-down
falls back to local, logged clearly, not silently or fatally.
"""
import json
import time
import unittest.mock as mock

import pytest

from core import cache_backend as cb


# --- LocalExactCache ---------------------------------------------------------

@pytest.fixture
def local_cache(tmp_path):
    return cb.LocalExactCache(path=str(tmp_path / "exact.json"), max_size=3)


def test_local_set_then_get_round_trips(local_cache):
    local_cache.set("hash1", {"risk": "HIGH", "score": 0.9})
    entry = local_cache.get("hash1")
    assert entry["risk"] == "HIGH"
    assert entry["score"] == 0.9


def test_local_get_missing_key_is_none(local_cache):
    assert local_cache.get("nope") is None


def test_local_ttl_expiry(local_cache):
    local_cache.set("hash1", {"risk": "LOW", "score": 0.1}, ttl_seconds=0)
    time.sleep(0.01)
    assert local_cache.get("hash1") is None


def test_local_lru_eviction_at_max_size(local_cache):
    local_cache.set("a", {"risk": "LOW", "score": 0.1})
    local_cache.set("b", {"risk": "LOW", "score": 0.1})
    local_cache.set("c", {"risk": "LOW", "score": 0.1})
    local_cache.set("d", {"risk": "LOW", "score": 0.1})  # evicts "a"

    assert local_cache.get("a") is None
    assert local_cache.get("d") is not None


def test_local_delete(local_cache):
    local_cache.set("hash1", {"risk": "HIGH", "score": 0.9})
    local_cache.delete("hash1")
    assert local_cache.get("hash1") is None


def test_local_flush_clears_everything(local_cache):
    local_cache.set("a", {"risk": "LOW", "score": 0.1})
    local_cache.flush()
    assert local_cache.get("a") is None


def test_local_persists_across_instances(tmp_path):
    path = str(tmp_path / "exact.json")
    c1 = cb.LocalExactCache(path=path)
    c1.set("hash1", {"risk": "HIGH", "score": 0.9})
    c1._save_thread.join(timeout=2)  # ensure the async write completed

    c2 = cb.LocalExactCache(path=path)
    assert c2.get("hash1")["risk"] == "HIGH"


# --- RedisExactCache (mocked client, no real server needed) -----------------

@pytest.fixture
def fake_redis():
    return mock.MagicMock()


def test_redis_get_deserialises_json(fake_redis):
    fake_redis.get.return_value = json.dumps({"risk": "HIGH", "score": 0.9}).encode()
    backend = cb.RedisExactCache(fake_redis)

    entry = backend.get("hash1")

    assert entry == {"risk": "HIGH", "score": 0.9}
    fake_redis.get.assert_called_once_with("gatekeeper:cache:hash1")


def test_redis_get_miss_returns_none(fake_redis):
    fake_redis.get.return_value = None
    assert cb.RedisExactCache(fake_redis).get("hash1") is None


def test_redis_get_corrupt_value_is_a_miss_not_a_crash(fake_redis):
    fake_redis.get.return_value = b"not valid json"
    assert cb.RedisExactCache(fake_redis).get("hash1") is None


def test_redis_set_uses_setex_with_ttl(fake_redis):
    backend = cb.RedisExactCache(fake_redis)
    backend.set("hash1", {"risk": "HIGH", "score": 0.9}, ttl_seconds=3600)

    fake_redis.setex.assert_called_once()
    args, _ = fake_redis.setex.call_args
    assert args[0] == "gatekeeper:cache:hash1"
    assert args[1] == 3600
    assert json.loads(args[2]) == {"risk": "HIGH", "score": 0.9}


def test_redis_flush_only_touches_this_apps_keys(fake_redis):
    fake_redis.scan_iter.return_value = ["gatekeeper:cache:a", "gatekeeper:cache:b"]
    cb.RedisExactCache(fake_redis).flush()

    fake_redis.scan_iter.assert_called_once_with("gatekeeper:cache:*")
    assert fake_redis.delete.call_count == 2
    # Never a blanket FLUSHDB.
    assert not hasattr(fake_redis, "flushdb") or not fake_redis.flushdb.called


# --- build_exact_cache_backend: selection and fallback -----------------------

def test_no_redis_url_selects_local(monkeypatch, tmp_path):
    monkeypatch.delenv("REDIS_URL", raising=False)
    backend = cb.build_exact_cache_backend()
    assert isinstance(backend, cb.LocalExactCache)


def test_redis_url_set_and_reachable_selects_redis(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.return_value = True

    with mock.patch("redis.from_url", return_value=fake_client):
        backend = cb.build_exact_cache_backend()

    assert isinstance(backend, cb.RedisExactCache)


def test_redis_url_set_but_unreachable_falls_back_to_local(monkeypatch):
    """THE CONTRACT THAT MATTERS MOST: a broken Redis must not crash startup
    or silently produce a non-functional cache — it must fall back cleanly."""
    monkeypatch.setenv("REDIS_URL", "redis://unreachable-host:6379/0")

    with mock.patch("redis.from_url", side_effect=ConnectionError("refused")):
        backend = cb.build_exact_cache_backend()

    assert isinstance(backend, cb.LocalExactCache)


def test_redis_ping_failure_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.side_effect = TimeoutError("no response")

    with mock.patch("redis.from_url", return_value=fake_client):
        backend = cb.build_exact_cache_backend()

    assert isinstance(backend, cb.LocalExactCache)
