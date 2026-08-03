"""
Pluggable exact-match cache backend.

WHY THIS EXISTS: the semantic cache's exact-hash tier (core/cache.py) is the
one tier this project makes an unconditional correctness claim about (see
FAISSCache.lookup's docstring — it is what §1i's cache fix relies on). Until
this file, it lived only in a process-local dict plus a JSON file, meaning
every gateway instance in a horizontally-scaled deployment had its OWN cache.
A verdict learned by one instance — including an async Llama Guard escalation
(core/risk.py's llama_guard_async_confirmation) — was invisible to every other
instance. That single-node coupling was the largest concrete item in the
infra gap versus a production guardrail service: it is the reason this system
could not run more than one replica and still behave consistently.

Redis is the natural fit for this tier specifically: it is a plain
key -> JSON-blob store with a TTL, which is exactly what the exact-match tier
already is. The FUZZY FAISS tier (core/cache.py's tier 2) deliberately stays
per-instance — it is already the tier this project does NOT make an
unconditional safety claim about (see CACHE_SIMILARITY_THRESHOLD's own
documentation), so per-instance eventual consistency there does not weaken any
existing guarantee. A truly distributed vector index (e.g. Redis's vector
search, or a real vector DB) is a larger undertaking, deliberately out of
scope here.

Backend selection happens ONCE at process startup, not per request. A
request-time Redis hiccup is handled by try/except around individual calls in
core/cache.py, not by silently re-routing to a different backend mid-flight —
that would split traffic across two inconsistent stores with no one noticing.
"""
import json
import os
import threading
import time
from collections import OrderedDict

from core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TTL_SECONDS = 86400  # 24 hours


class ExactCacheBackend:
    """Interface: get/set/delete a JSON-serialisable entry by prompt hash."""

    def get(self, key):
        raise NotImplementedError

    def set(self, key, entry, ttl_seconds=DEFAULT_TTL_SECONDS):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


class LocalExactCache(ExactCacheBackend):
    """
    Process-local, file-persisted backend — the pre-existing behaviour,
    extracted essentially unchanged so single-node deployments (and the test
    suite, which never sets REDIS_URL) are unaffected by this refactor.
    """

    def __init__(self, path="semantic_cache_exact.json", max_size=5000):
        self.path = path
        self.max_size = max_size
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self._save_thread = None
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            now = time.time()
            for key, entry in raw.items():
                if now - entry.get("timestamp_epoch", 0) < entry.get("ttl_seconds", DEFAULT_TTL_SECONDS):
                    self._data[key] = entry
        except Exception as e:
            logger.error(f"Local exact-cache load error: {e}")

    def _save_async(self):
        def _write():
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(dict(self._data), f)
            except Exception as e:
                logger.error(f"Local exact-cache save error: {e}")

        if self._save_thread is None or not self._save_thread.is_alive():
            self._save_thread = threading.Thread(target=_write)
            self._save_thread.start()

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if time.time() - entry.get("timestamp_epoch", 0) >= entry.get("ttl_seconds", DEFAULT_TTL_SECONDS):
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return dict(entry)

    def set(self, key, entry, ttl_seconds=DEFAULT_TTL_SECONDS):
        with self._lock:
            entry = dict(entry)
            entry["timestamp_epoch"] = time.time()
            entry["ttl_seconds"] = ttl_seconds
            if key in self._data:
                del self._data[key]
            elif len(self._data) >= self.max_size:
                self._data.popitem(last=False)
            self._data[key] = entry
        self._save_async()

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)
        self._save_async()

    def flush(self):
        with self._lock:
            self._data = OrderedDict()
        self._save_async()


class RedisExactCache(ExactCacheBackend):
    """Shared backend: every gateway instance reads and writes the SAME
    store. TTL is native to Redis (SETEX), not re-implemented by hand."""

    KEY_PREFIX = "gatekeeper:cache:"

    def __init__(self, client):
        self._client = client

    def get(self, key):
        raw = self._client.get(self.KEY_PREFIX + key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set(self, key, entry, ttl_seconds=DEFAULT_TTL_SECONDS):
        self._client.setex(self.KEY_PREFIX + key, ttl_seconds, json.dumps(entry))

    def delete(self, key):
        self._client.delete(self.KEY_PREFIX + key)

    def flush(self):
        # Deliberately scoped to this app's own keys — never FLUSHDB on a
        # shared Redis instance that might hold other applications' data.
        for k in self._client.scan_iter(self.KEY_PREFIX + "*"):
            self._client.delete(k)


def build_exact_cache_backend(max_size=5000):
    """
    Selects Redis if REDIS_URL is set AND reachable right now, else falls
    back to the local file-backed store — logged clearly either way, so a
    deployment that meant to run distributed but has a silently-broken Redis
    connection sees it in the startup log, not via mysteriously inconsistent
    verdicts across instances later.

    `max_size` only applies to the local fallback — an in-process dict needs
    an explicit cap. Redis is left to its own TTL-based expiry; imposing a
    client-side entry count on a shared store doesn't map cleanly onto
    multiple instances writing to it concurrently, and Redis's own
    maxmemory-policy is the right lever for that if ever needed.
    """
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        logger.info("REDIS_URL not set; using local file-backed cache (single-node only).")
        return LocalExactCache(max_size=max_size)

    try:
        import redis  # noqa: PLC0415
        client = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        safe_url = redis_url.split("@")[-1]  # drop any embedded credentials before logging
        logger.info(f"Connected to Redis at {safe_url}; cache is now shared across instances.")
        return RedisExactCache(client)
    except Exception as e:
        logger.error(
            f"REDIS_URL is set but Redis is unreachable ({type(e).__name__}: {e}); "
            f"falling back to the local file-backed cache. Verdicts will NOT be "
            f"shared across gateway instances until this is fixed."
        )
        return LocalExactCache()
