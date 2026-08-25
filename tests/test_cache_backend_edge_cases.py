"""
Additional edge-case coverage for core/cache_backend.py that
tests/test_cache_backend.py does not exercise: corrupted/partially-expired
on-disk state at load time, key-overwrite semantics, redis-package-missing
fallback, credential redaction in the startup log, and small-boundary
max_size behaviour.

Kept in a separate file per the test-expansion plan, to avoid touching the
existing, already-thorough test_cache_backend.py.
"""
import json
import time
import unittest.mock as mock


from core import cache_backend as cb


# --- LocalExactCache: on-disk load edge cases --------------------------------

def test_load_skips_expired_entries_present_on_disk(tmp_path):
    """_load() filters entries whose TTL had already elapsed before this
    process even started -- not just ones that expire while running."""
    path = tmp_path / "exact.json"
    now = time.time()
    raw = {
        "expired": {"risk": "HIGH", "score": 0.9, "timestamp_epoch": now - 100, "ttl_seconds": 10},
        "fresh": {"risk": "LOW", "score": 0.1, "timestamp_epoch": now, "ttl_seconds": 86400},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    cache = cb.LocalExactCache(path=str(path))

    assert cache.get("expired") is None
    entry = cache.get("fresh")
    assert entry is not None and entry["risk"] == "LOW"


def test_load_with_corrupt_json_file_starts_empty_not_crashing(tmp_path):
    """A corrupted cache file on disk must not prevent the process from
    starting -- _load() logs and continues with an empty cache."""
    path = tmp_path / "exact.json"
    path.write_text("{not valid json!!", encoding="utf-8")

    cache = cb.LocalExactCache(path=str(path))

    assert cache.get("anything") is None
    # And it must still be a fully working cache afterward.
    cache.set("k", {"risk": "LOW", "score": 0.1})
    assert cache.get("k")["risk"] == "LOW"


def test_load_missing_file_starts_empty(tmp_path):
    path = tmp_path / "does_not_exist.json"
    cache = cb.LocalExactCache(path=str(path))
    assert cache.get("anything") is None


# --- LocalExactCache: set/overwrite/delete semantics -------------------------

def test_set_overwriting_existing_key_does_not_duplicate_and_updates_value(tmp_path):
    cache = cb.LocalExactCache(path=str(tmp_path / "exact.json"), max_size=5)
    cache.set("hash1", {"risk": "LOW", "score": 0.1})
    cache.set("hash1", {"risk": "HIGH", "score": 0.9})

    assert len(cache._data) == 1
    entry = cache.get("hash1")
    assert entry["risk"] == "HIGH"
    assert entry["score"] == 0.9


def test_set_overwrite_moves_key_to_most_recently_used_end(tmp_path):
    """Re-setting an existing key must refresh its LRU position, protecting
    it from the next eviction just like a fresh insert would."""
    cache = cb.LocalExactCache(path=str(tmp_path / "exact.json"), max_size=2)
    cache.set("a", {"risk": "LOW", "score": 0.1})
    cache.set("b", {"risk": "LOW", "score": 0.1})
    cache.set("a", {"risk": "LOW", "score": 0.2})  # refresh "a" -> now MRU
    cache.set("c", {"risk": "LOW", "score": 0.3})  # must evict "b", not "a"

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None


def test_delete_of_missing_key_does_not_raise(tmp_path):
    cache = cb.LocalExactCache(path=str(tmp_path / "exact.json"))
    cache.delete("never-existed")  # must not raise
    assert cache.get("never-existed") is None


def test_get_moves_entry_to_most_recently_used_end(tmp_path):
    """A plain get() (not just set()) must count as usage for LRU purposes."""
    cache = cb.LocalExactCache(path=str(tmp_path / "exact.json"), max_size=2)
    cache.set("a", {"risk": "LOW", "score": 0.1})
    cache.set("b", {"risk": "LOW", "score": 0.1})
    cache.get("a")  # touch "a" -> now MRU, "b" becomes LRU
    cache.set("c", {"risk": "LOW", "score": 0.1})  # must evict "b"

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None


def test_max_size_of_one_still_allows_one_entry_and_evicts_on_second(tmp_path):
    cache = cb.LocalExactCache(path=str(tmp_path / "exact.json"), max_size=1)
    cache.set("a", {"risk": "LOW", "score": 0.1})
    assert cache.get("a") is not None

    cache.set("b", {"risk": "HIGH", "score": 0.9})
    assert cache.get("a") is None
    assert cache.get("b")["risk"] == "HIGH"


def test_get_returns_a_copy_not_a_live_reference(tmp_path):
    """Mutating the dict returned by get() must not corrupt internal state."""
    cache = cb.LocalExactCache(path=str(tmp_path / "exact.json"))
    cache.set("a", {"risk": "LOW", "score": 0.1})
    entry = cache.get("a")
    entry["risk"] = "TAMPERED"

    fresh = cache.get("a")
    assert fresh["risk"] == "LOW"


# --- RedisExactCache: additional behaviour -----------------------------------

def test_redis_key_prefix_isolates_from_raw_key_collisions():
    """Two logically-different app keys that happen to collide with the raw
    KEY_PREFIX string itself are still addressed distinctly by the backend
    (the prefix is prepended, not substituted)."""
    fake_redis = mock.MagicMock()
    backend = cb.RedisExactCache(fake_redis)
    backend.get("gatekeeper:cache:already-prefixed-looking-key")
    fake_redis.get.assert_called_once_with(
        "gatekeeper:cache:gatekeeper:cache:already-prefixed-looking-key"
    )


def test_redis_set_serialises_complex_nested_entry():
    fake_redis = mock.MagicMock()
    backend = cb.RedisExactCache(fake_redis)
    complex_entry = {
        "risk": "HIGH",
        "score": 0.87,
        "reasons": ["pattern_match", "llm_judge"],
        "metadata": {"model": "llama-guard-8b", "nested": {"a": 1}},
    }
    backend.set("hash1", complex_entry)

    args, _ = fake_redis.setex.call_args
    assert json.loads(args[2]) == complex_entry


def test_redis_flush_with_no_matching_keys_deletes_nothing():
    fake_redis = mock.MagicMock()
    fake_redis.scan_iter.return_value = []
    cb.RedisExactCache(fake_redis).flush()
    fake_redis.delete.assert_not_called()


# --- build_exact_cache_backend: additional selection paths -------------------

def test_redis_package_not_installed_falls_back_to_local(monkeypatch):
    """If REDIS_URL is set but the `redis` package itself can't be imported
    (not installed), selection must still fall back cleanly rather than
    raising an unhandled ImportError."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("No module named 'redis'")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        backend = cb.build_exact_cache_backend()

    assert isinstance(backend, cb.LocalExactCache)


def test_max_size_is_passed_through_to_local_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("REDIS_URL", raising=False)
    backend = cb.build_exact_cache_backend(max_size=42)
    assert isinstance(backend, cb.LocalExactCache)
    assert backend.max_size == 42


def test_startup_log_redacts_embedded_redis_credentials(monkeypatch, caplog):
    """A Redis URL with embedded credentials must never appear verbatim in
    the startup log -- only the host/port portion after the '@'."""
    monkeypatch.setenv("REDIS_URL", "redis://user:supersecret@localhost:6379/0")
    fake_client = mock.MagicMock()
    fake_client.ping.return_value = True

    with mock.patch("redis.from_url", return_value=fake_client):
        with caplog.at_level("INFO"):
            cb.build_exact_cache_backend()

    assert "supersecret" not in caplog.text
    assert "localhost:6379/0" in caplog.text


def test_unreachable_redis_error_log_does_not_leak_credentials(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://user:supersecret@unreachable-host:6379/0")

    with mock.patch("redis.from_url", side_effect=ConnectionError("refused")):
        with caplog.at_level("ERROR"):
            backend = cb.build_exact_cache_backend()

    assert isinstance(backend, cb.LocalExactCache)
    # The error branch logs the exception message, not the URL, so the
    # credential must not appear anywhere in the captured log output.
    assert "supersecret" not in caplog.text
