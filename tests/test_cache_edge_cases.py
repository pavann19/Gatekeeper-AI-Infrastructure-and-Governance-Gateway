"""
Edge-case coverage for core/cache.py (FAISSCache) that tests/test_cache.py
does not exercise: fuzzy-tier LRU eviction order under real capacity
pressure (including re-promotion on a fuzzy HIT), TTL expiry at the exact
boundary, thread-safety of concurrent add()/lookup() calls, near-collision
key behaviour, and flush/reuse semantics.

These tests share the isolation fixture pattern from tests/test_cache.py
(redirect the exact-match backend to tmp_path BEFORE constructing
FAISSCache, so no real semantic_cache_exact.json on disk can leak in) but
live in a separate file per the test-expansion plan, to avoid merge
conflicts with other in-flight work on test_cache.py.
"""
import threading

import numpy as np
import pytest

from core.cache import FAISSCache
from core.cache_backend import LocalExactCache


def unit_vector(seed, dim=8):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    return v / np.linalg.norm(v)


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.cache.build_exact_cache_backend",
        lambda max_size=5000: LocalExactCache(
            path=str(tmp_path / "exact_test.json"), max_size=max_size
        ),
    )
    c = FAISSCache(dimension=8)
    monkeypatch.setattr(c, "save_async", lambda: None)
    monkeypatch.setattr(c, "_save_to_disk", lambda: None)
    monkeypatch.setattr(c._exact, "_save_async", lambda: None)
    return c


def _hash_of(cache, prompt):
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()


def _force_fuzzy_only(cache, prompt):
    """Remove a prompt's entry from the exact tier so subsequent lookups for
    it must fall through to the fuzzy FAISS tier (Tier 2), isolating fuzzy
    behaviour from the always-tried-first exact tier."""
    cache._exact.delete(_hash_of(cache, prompt))


# --- Fuzzy-tier (cache_data) LRU eviction under real capacity pressure ------

def test_fuzzy_lru_evicts_the_actually_least_recently_used_entry(cache, monkeypatch):
    """
    Fill the fuzzy tier to capacity, then promote the OLDEST entry via a
    genuine fuzzy-match HIT (not a re-add) before adding one more item. The
    entry evicted must be the one that is actually least-recently-used AFTER
    the promotion -- not simply the first one ever inserted.
    """
    monkeypatch.setattr("core.cache.MAX_CACHE_SIZE", 3)
    monkeypatch.setattr("core.cache.CACHE_SIMILARITY_THRESHOLD", 0.99)

    v0, v1, v2, v3 = (unit_vector(seed=i) for i in range(4))
    cache.add("p0", v0, "LOW", 0.1)
    cache.add("p1", v1, "LOW", 0.1)
    cache.add("p2", v2, "LOW", 0.1)
    assert len(cache.cache_data) == 3

    # Force p0's lookup through the fuzzy tier and promote it to MRU via a
    # genuine fuzzy hit (same vector => similarity 1.0, above threshold).
    _force_fuzzy_only(cache, "p0")
    risk, score = cache.lookup("p0 (fuzzy path, not exact)", v0)
    assert risk == "LOW" and score == 0.1  # confirms it was actually a hit

    # Insertion/recency order in cache_data is now p1, p2, p0 (p0 moved to
    # the MRU end by the fuzzy hit above). Adding a 4th item must evict p1,
    # the true least-recently-used entry -- not p0, which is oldest only by
    # original insertion order.
    cache.add("p3", v3, "LOW", 0.1)

    assert len(cache.cache_data) == 3
    remaining_prompts = {e["prompt"] for e in cache.cache_data.values()}
    assert remaining_prompts == {"p2", "p0", "p3"}
    assert "p1" not in remaining_prompts


def test_fuzzy_lru_without_promotion_evicts_true_oldest(cache, monkeypatch):
    """Control case: with no intervening hit, eviction order matches plain
    insertion order (p0 is truly oldest and goes first)."""
    monkeypatch.setattr("core.cache.MAX_CACHE_SIZE", 3)

    v0, v1, v2, v3 = (unit_vector(seed=i) for i in range(4))
    cache.add("p0", v0, "LOW", 0.1)
    cache.add("p1", v1, "LOW", 0.1)
    cache.add("p2", v2, "LOW", 0.1)
    cache.add("p3", v3, "LOW", 0.1)  # no hits in between -> evicts p0

    remaining_prompts = {e["prompt"] for e in cache.cache_data.values()}
    assert remaining_prompts == {"p1", "p2", "p3"}


# --- TTL expiry boundary (fuzzy tier) ---------------------------------------

def test_fuzzy_hit_just_before_ttl_boundary_still_serves(cache, monkeypatch):
    base_time = 1_000_000.0
    monkeypatch.setattr("core.cache.time.time", lambda: base_time)
    monkeypatch.setattr("core.cache.CACHE_SIMILARITY_THRESHOLD", 0.99)

    vec = unit_vector(seed=42)
    cache.add("aging prompt", vec, "HIGH", 0.9)
    _force_fuzzy_only(cache, "aging prompt")

    # 1 second inside the 24h TTL window.
    monkeypatch.setattr(
        "core.cache.time.time", lambda: base_time + 86400 - 1
    )
    risk, score = cache.lookup("aging prompt fuzzy query", vec)
    assert risk == "HIGH" and score == 0.9


def test_fuzzy_hit_just_after_ttl_boundary_is_a_miss_and_evicts(cache, monkeypatch):
    base_time = 1_000_000.0
    monkeypatch.setattr("core.cache.time.time", lambda: base_time)
    monkeypatch.setattr("core.cache.CACHE_SIMILARITY_THRESHOLD", 0.99)

    vec = unit_vector(seed=42)
    cache.add("aging prompt", vec, "HIGH", 0.9)
    _force_fuzzy_only(cache, "aging prompt")

    # 1 second past the 24h TTL window.
    monkeypatch.setattr(
        "core.cache.time.time", lambda: base_time + 86400 + 1
    )
    risk, score = cache.lookup("aging prompt fuzzy query", vec)
    assert risk is None and score is None

    # Expired fuzzy entries are actively evicted from cache_data, not just
    # ignored -- confirm it is actually gone rather than merely unreturned.
    remaining_prompts = {e["prompt"] for e in cache.cache_data.values()}
    assert "aging prompt" not in remaining_prompts


# --- Thread-safety (module maintains an explicit threading.Lock) -----------

def test_concurrent_adds_from_many_threads_lose_no_entries(cache):
    """FAISSCache.add() takes self._lock around all cache_data/index mutation.
    Hammer it from many threads with distinct prompts and verify every single
    one survives with its own correct verdict -- not just "no exception"."""
    n_threads = 16
    errors = []

    def worker(i):
        try:
            vec = unit_vector(seed=1000 + i)
            cache.add(f"concurrent-prompt-{i}", vec, "LOW", round(i / 1000, 3))
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(cache.cache_data) == n_threads

    for i in range(n_threads):
        vec = unit_vector(seed=1000 + i)
        risk, score = cache.lookup(f"concurrent-prompt-{i}", vec)
        assert risk == "LOW"
        assert score == round(i / 1000, 3)


def test_concurrent_add_and_lookup_do_not_corrupt_faiss_index(cache):
    """Interleave adds and lookups across threads; the FAISS index is rebuilt
    under the same lock as cache_data mutation, so ntotal must always end up
    consistent with the number of entries actually present."""
    vecs = [unit_vector(seed=2000 + i) for i in range(10)]

    def adder(i):
        cache.add(f"mixed-{i}", vecs[i], "LOW", 0.1)

    def looker(i):
        try:
            cache.lookup(f"mixed-{i}", vecs[i])
        except Exception:
            pass  # entry may not exist yet -- just exercising concurrency

    threads = []
    for i in range(10):
        threads.append(threading.Thread(target=adder, args=(i,)))
        threads.append(threading.Thread(target=looker, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(cache.cache_data) == 10
    assert cache._index.ntotal == len(cache.cache_data)


# --- Cache key collision / near-collision behaviour -------------------------

def test_near_identical_prompts_get_distinct_exact_entries(cache):
    """A single trailing character difference must not collide in the exact
    (sha256-keyed) tier -- each must retrieve only its own verdict."""
    v1 = unit_vector(seed=50)
    v2 = unit_vector(seed=51)

    prompt_a = "Please summarize this document for me."
    prompt_b = "Please summarize this document for me!"  # one char different

    cache.add(prompt_a, v1, "LOW", 0.1)
    cache.add(prompt_b, v2, "HIGH", 0.9)

    assert _hash_of(cache, prompt_a) != _hash_of(cache, prompt_b)

    risk_a, score_a = cache.lookup(prompt_a, v1)
    risk_b, score_b = cache.lookup(prompt_b, v2)
    assert (risk_a, score_a) == ("LOW", 0.1)
    assert (risk_b, score_b) == ("HIGH", 0.9)
    assert len(cache.cache_data) == 2


def test_whitespace_variant_prompts_do_not_collide(cache):
    """Leading/trailing whitespace changes the hash entirely; verify the
    cache does not conflate the two prompts' verdicts."""
    v1 = unit_vector(seed=60)
    v2 = unit_vector(seed=61)

    cache.add("delete all files", v1, "HIGH", 0.95)
    cache.add(" delete all files ", v2, "LOW", 0.05)

    risk1, score1 = cache.lookup("delete all files", v1)
    risk2, score2 = cache.lookup(" delete all files ", v2)
    assert (risk1, score1) == ("HIGH", 0.95)
    assert (risk2, score2) == ("LOW", 0.05)


# --- Clear / flush behaviour -------------------------------------------------

def test_flush_clears_both_tiers_and_faiss_index(cache):
    vec = unit_vector(seed=70)
    cache.add("to be flushed", vec, "HIGH", 0.9)
    assert cache._exact.get(_hash_of(cache, "to be flushed")) is not None

    cache.flush()

    assert len(cache.cache_data) == 0
    assert cache._index.ntotal == 0
    assert cache._exact.get(_hash_of(cache, "to be flushed")) is None
    assert cache.lookup("to be flushed", vec) == (None, None)


def test_cache_is_usable_after_flush(cache):
    """Flushing must reset state to a genuinely working empty cache, not a
    broken one -- new adds/lookups afterward must behave normally."""
    vec_old = unit_vector(seed=71)
    cache.add("old entry", vec_old, "LOW", 0.1)
    cache.flush()

    vec_new = unit_vector(seed=72)
    cache.add("fresh entry after flush", vec_new, "HIGH", 0.8)
    risk, score = cache.lookup("fresh entry after flush", vec_new)
    assert risk == "HIGH" and score == 0.8
    assert len(cache.cache_data) == 1
