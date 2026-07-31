import json
import os
import hashlib
import time
from collections import OrderedDict
from datetime import datetime, timezone
import threading
from core.config import CACHE_SIMILARITY_THRESHOLD
from core.logger import get_logger
from core.embeddings import cosine_similarity
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
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if hasattr(vector, 'tolist'): 
                vector_list = vector.tolist()
            else:
                vector_list = vector
                
            entry = {
                "prompt": prompt,
                "vector": vector_list,
                "risk": risk,
                "score": score,
                "prompt_hash": prompt_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timestamp_epoch": time.time(),
                "source": source
            }
            
            # LRU Eviction
            if prompt_hash in self.cache_data:
                del self.cache_data[prompt_hash]
            elif len(self.cache_data) >= MAX_CACHE_SIZE:
                self.cache_data.popitem(last=False)
                
            self.cache_data[prompt_hash] = entry
            
            # Since we modify the OrderedDict, we must rebuild FAISS
            self._rebuild_faiss()
            
        # Trigger non-blocking save
        self.save_async()

    def lookup(self, prompt, query_vec):
        """
        Two-tier lookup: an exact prompt-hash match first, a fuzzy FAISS
        similarity match second. See module docstring for why the tiers exist
        and are not interchangeable.
        """
        if not self.cache_data:
            return None, None

        with self._lock:
            # ---- TIER 1: EXACT MATCH ----
            # O(1), zero collision risk by construction — the same query text
            # can only ever retrieve the verdict it (or an earlier identical
            # query) actually received. This is what almost all real cache
            # value comes from (the same question asked repeatedly) and it is
            # the only tier this class can make an unconditional safety claim
            # about.
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            entry = self.cache_data.get(prompt_hash)
            if entry is not None:
                if time.time() - entry.get("timestamp_epoch", 0) < TTL_SECONDS:
                    self.cache_data.move_to_end(prompt_hash)
                    return entry.get("risk"), entry.get("score")
                else:
                    del self.cache_data[prompt_hash]
                    self._rebuild_faiss()
                    # Fall through to fuzzy match below — the exact entry is
                    # gone, but a similar one may still be usable.

            # ---- TIER 2: FUZZY SIMILARITY MATCH ----
            # measured (scripts/diagnose_cache_threshold.py) against
            # deepset/prompt-injections: at the OLD default of 0.95, 9.1% of
            # near-duplicate pairs above threshold had OPPOSITE ground-truth
            # labels (the dataset mutates a benign wrapper by inserting an
            # injection payload — nearly identical surface text, opposite
            # verdict), and at 0.98 that rate was 50%. Sentence-embedding
            # cosine similarity measures bulk topical content, not the
            # presence of a short adversarial clause, so it cannot safely
            # stand in for a security decision at any threshold low enough to
            # matter. CACHE_SIMILARITY_THRESHOLD's default was raised to 0.99
            # (zero unsafe pairs observed at or above that value in the same
            # measurement) as a materially safer margin — NOT a guarantee.
            # This tier exists for genuine near-duplicates; if this residual
            # risk is unacceptable for a deployment, disable it entirely by
            # setting the threshold to 1.0, which leaves exact-match caching
            # (tier 1) fully intact.
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
                if time.time() - entry.get("timestamp_epoch", 0) < TTL_SECONDS:
                    # LRU update
                    prompt_hash = entry["prompt_hash"]
                    self.cache_data.move_to_end(prompt_hash)
                    return entry.get("risk"), entry.get("score")
                else:
                    # Expired
                    del self.cache_data[entry["prompt_hash"]]
                    self._rebuild_faiss()

        return None, None
        
    def flush(self):
        with self._lock:
            self.cache_data = OrderedDict()
            self._rebuild_faiss()
            self._save_to_disk()
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
