from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import json
import sys
import os
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import time
from collections import OrderedDict
from datetime import datetime, timezone
import threading
from core.config import CACHE_SIMILARITY_THRESHOLD
from core.cache_backend import build_exact_cache_backend
from core.logger import get_logger
import numpy as np

logger = get_logger(__name__)

CACHE_FILE = "semantic_cache.json"
MAX_CACHE_SIZE = 5000
TTL_SECONDS = 86400  # 24 hours

class FAISSCache:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.cache_data = OrderedDict()
        self._index = None
        self._lock = threading.Lock()
        self._save_thread = None
        # Exact-match tier (Tier 1 in lookup()) is delegated to a pluggable
        # backend — Redis when REDIS_URL is configured and reachable, else a
        # local file-backed store. See core/cache_backend.py for why this
        # tier specifically is the one worth making shared across instances.
        # cache_data (above) remains the LOCAL, per-instance store backing
        # the fuzzy FAISS index (Tier 2) regardless of which exact backend is
        # active.
        self._exact = build_exact_cache_backend(max_size=MAX_CACHE_SIZE)
        
    def _get_index(self):
        if self._index is None:
            import faiss  # noqa: PLC0415
            # We use IndexIDMap to allow removing items, or we can just rebuild it.
            # Rebuilding 5000 items is virtually instantaneous, so we just rebuild it.
            self._index = faiss.IndexFlatIP(self.dimension)
            self._rebuild_faiss()
        return self._index

    def _rebuild_faiss(self):
        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)
        if not self.cache_data:
            return
            
        vectors = []
        for key, entry in self.cache_data.items():
            vectors.append(entry["vector"])
            
        vec_matrix = np.array(vectors).astype('float32')
        faiss.normalize_L2(vec_matrix)
        self._index.add(vec_matrix)

    def load(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                
                now = time.time()
                for entry in data:
                    # Apply TTL on load
                    entry_time = entry.get("timestamp_epoch", 0)
                    if now - entry_time < TTL_SECONDS:
                        self.cache_data[entry["prompt_hash"]] = entry
                
                logger.info(f"Semantic Cache Loaded ({len(self.cache_data)} active entries)")
            except Exception as e:
                logger.error(f"Cache Load Error: {e}")
                self.cache_data = OrderedDict()
        else:
            logger.info("Semantic Cache Initialized (Empty)")

    def _save_to_disk(self):
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(list(self.cache_data.values()), f)
        except Exception as e:
            logger.error(f"Cache Save Error: {e}")

    def save_async(self):
        """Saves cache to disk in a background thread to prevent blocking the event loop."""
        if self._save_thread is None or not self._save_thread.is_alive():
            self._save_thread = threading.Thread(target=self._save_to_disk)
            self._save_thread.start()

    def add(self, prompt, vector, risk, score, source="unknown"):
        with self._lock:
            prompt_hash = _security_key(prompt)
            if hasattr(vector, 'tolist'):
                vector_list = vector.tolist()
            else:
                vector_list = vector

            entry = {
                "prompt": prompt,
                "security_scope": _CACHE_SCOPE.get(),
                "vector": vector_list,
                "risk": risk,
                "score": score,
                "prompt_hash": prompt_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timestamp_epoch": time.time(),
                "source": source
            }

            # LRU Eviction (local fuzzy-index copy only — the exact backend
            # manages its own eviction/TTL independently, see cache_backend.py)
            existing = self.cache_data.get(prompt_hash)
            if existing and existing.get("risk") == "HIGH" and time.time() - existing.get("timestamp_epoch", 0) < TTL_SECONDS:
                entry = existing
            if prompt_hash in self.cache_data:
                del self.cache_data[prompt_hash]
            elif len(self.cache_data) >= MAX_CACHE_SIZE:
                self.cache_data.popitem(last=False)

            self.cache_data[prompt_hash] = entry

            # Since we modify the OrderedDict, we must rebuild FAISS
            self._rebuild_faiss()

        # Write-through to the exact-match backend OUTSIDE the lock — this
        # may be a network call (Redis) and must not hold up the fuzzy-index
        # rebuild above for other threads. A failure here is logged, not
        # raised: the local fuzzy tier above already has the entry, so this
        # instance still benefits even if the shared write failed.
        try:
            self._exact.set(prompt_hash, entry, ttl_seconds=TTL_SECONDS)
        except Exception as e:
            logger.error(f"Exact-cache backend write failed ({type(e).__name__}: {e}); "
                        f"this instance's local cache still has the entry.")

        # Trigger non-blocking save of the local fuzzy-index snapshot
        self.save_async()

    def lookup(self, prompt, query_vec):
        """
        Two-tier lookup: an exact prompt-hash match first, a fuzzy FAISS
        similarity match second. See module docstring for why the tiers exist
        and are not interchangeable.
        """
        # ---- TIER 1: EXACT MATCH ----
        # Versioned exact identity includes tenant, raw-input hash and clean text.
        # Only HIGH short-circuits assessment. LOW/MEDIUM always run fresh checks.
        # Backend calls stay outside the FAISS lock because Redis may block.
        prompt_hash = _security_key(prompt)
        try:
            entry = self._exact.get(prompt_hash)
        except Exception as e:
            logger.error(f"Exact-cache backend read failed ({type(e).__name__}: {e}); "
                        f"treating as a miss and falling through to fuzzy match.")
            entry = None
        if entry is not None and entry.get("risk") == "HIGH":
            return entry.get("risk"), entry.get("score")

        if not self.cache_data:
            return None, None

        with self._lock:
            # ---- TIER 2: FUZZY SIMILARITY MATCH ----
            # Similarity is not a permissive security verdict. Only a live HIGH
            # in the same security scope can escalate a fuzzy query.
            import faiss
            index = self._get_index()
            if index.ntotal == 0:
                return None, None

            if hasattr(query_vec, 'cpu'):
                vec = query_vec.cpu().numpy()
            else:
                vec = np.array(query_vec)

            vec = vec.reshape(1, -1).astype('float32')
            faiss.normalize_L2(vec)

            scores, indices = index.search(vec, 1)
            best_score = float(scores[0][0])
            best_idx = int(indices[0][0])

            if best_score > CACHE_SIMILARITY_THRESHOLD and best_idx != -1 and best_idx < len(self.cache_data):
                # We need to map FAISS index back to the OrderedDict
                # Since we rebuild FAISS directly from OrderedDict.values(), indices map 1:1
                entry = list(self.cache_data.values())[best_idx]

                # Check TTL
                if (entry.get("risk") == "HIGH" and entry.get("security_scope") == _CACHE_SCOPE.get()
                        and time.time() - entry.get("timestamp_epoch", 0) < TTL_SECONDS):
                    # LRU update
                    prompt_hash = entry["prompt_hash"]
                    self.cache_data.move_to_end(prompt_hash)
                    return entry.get("risk"), entry.get("score")
                elif time.time() - entry.get("timestamp_epoch", 0) >= TTL_SECONDS:
                    # Expired
                    del self.cache_data[entry["prompt_hash"]]
                    self._rebuild_faiss()

        return None, None
        
    def flush(self):
        with self._lock:
            self.cache_data = OrderedDict()
            self._rebuild_faiss()
            self._save_to_disk()
        try:
            self._exact.flush()
        except Exception as e:
            logger.error(f"Exact-cache backend flush failed: {type(e).__name__}: {e}")
        logger.info("Cache Flushed.")

# Singleton instance
_cache = FAISSCache()
_cache.load()

def save_cache_entry(prompt, vector, risk, score, source="unknown"):
    _cache.add(prompt, vector, risk, score, source)

def lookup_cache(prompt, new_vector):
    return _cache.lookup(prompt, new_vector)

def flush_cache():
    _cache.flush()


_CACHE_SCOPE = ContextVar("security_cache_scope", default="direct-v2")

@contextmanager
def security_cache_scope(raw_prompt, tenant):
    # Raw identity is hashed in memory; it is never persisted as text.
    digest = hashlib.sha256(b"gatekeeper-security-cache-v2")
    # Runtime interpretation can change scores with identical pinned weights.
    # A model-library change must create a new assessment version.
    for package in ("torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy", "tokenizers", "safetensors"):
        try:
            runtime_version = package_version(package)
        except PackageNotFoundError:
            runtime_version = "unavailable"
        digest.update(f"{package}={runtime_version}".encode())

    for directory in ("core", "policies", "models"):
        for path in sorted(Path(directory).glob("**/*")):
            if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".yml", ".toml"):
                digest.update(str(path).encode())
                digest.update(path.read_bytes())
    from core.config import settings
    digest.update(json.dumps(settings.model_dump(), sort_keys=True, default=str).encode())
    for key, value in settings.model_dump().items():
        if isinstance(value, str) and ("POLICY" in key or "MODEL" in key):
            path = Path(value)
            if path.is_file():
                digest.update(path.read_bytes())
    detectors = sys.modules.get("core.detectors")
    registry = getattr(detectors, "_REGISTRY", None) or {}
    state = {name: (getattr(detector, "_model", None) is not None,
                    type(getattr(detector, "_load_error", None)).__name__)
             for name, detector in registry.items()}
    digest.update(json.dumps(state, sort_keys=True).encode())
    scope = ":".join((digest.hexdigest(), hashlib.sha256(tenant.encode()).hexdigest(),
                      hashlib.sha256(raw_prompt.encode()).hexdigest()))
    token = _CACHE_SCOPE.set(scope)
    try:
        yield
    finally:
        _CACHE_SCOPE.reset(token)


def _security_key(prompt):
    return hashlib.sha256((_CACHE_SCOPE.get() + "\0" + prompt).encode()).hexdigest()
