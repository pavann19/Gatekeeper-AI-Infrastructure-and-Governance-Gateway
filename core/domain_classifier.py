# core/domain_classifier.py
import json
import os
from core.config import DOMAIN_THRESHOLD
from core.logger import get_logger

logger = get_logger(__name__)

DOMAIN_CORPUS_FILE = "policies/domain_corpus.json"

def _load_domain_corpus():
    """Loads domain corpus documents from external JSON file.
    Returns list of document strings, or empty list if file missing/invalid."""
    if not os.path.exists(DOMAIN_CORPUS_FILE):
        logger.warning(f"{DOMAIN_CORPUS_FILE} not found. Domain guardrail will default to allow.")
        return []
    try:
        with open(DOMAIN_CORPUS_FILE, "r") as f:
            data = json.load(f)
            return data.get("documents", [])
    except Exception as e:
        logger.warning(f"Failed to load domain corpus: {e}")
        return []

def _compute_centroid(documents):
    """Computes the average embedding (centroid) from a list of documents.
    Returns the centroid vector, or None if no documents provided."""
    from core.embeddings import get_embedding  # lazy import — avoids CI failure
    if not documents:
        return None
    embeddings = []
    for doc in documents:
        vec = get_embedding(doc)
        if vec is not None:
            if hasattr(vec, 'tolist'):
                vec = vec.tolist()
            embeddings.append(vec)
    if not embeddings:
        return None
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for vec in embeddings:
        for i in range(dim):
            centroid[i] += vec[i]
    for i in range(dim):
        centroid[i] /= len(embeddings)
    return centroid

# --- Lazy centroid: computed on first use, NOT at module import ---
_corpus_documents = None
_domain_centroid_cache = None
_centroid_initialized = False

def _get_domain_centroid():
    """Returns the domain centroid, computing it once on first call."""
    global _corpus_documents, _domain_centroid_cache, _centroid_initialized
    if not _centroid_initialized:
        _corpus_documents = _load_domain_corpus()
        _domain_centroid_cache = _compute_centroid(_corpus_documents)
        if _domain_centroid_cache is None:
            logger.warning("Domain centroid not computed. Domain check will default to allow.")
        _centroid_initialized = True
    return _domain_centroid_cache

def is_domain_aligned(prompt: str) -> tuple:
    """
    Checks if the prompt semantically aligns with allowed technical domains
    using centroid-based similarity.
    Returns: (bool, float) — (is_aligned, similarity_score)
    """
    from core.embeddings import get_embedding, cosine_similarity  # lazy import
    centroid = _get_domain_centroid()

    # If centroid unavailable, default to allow (avoid system crash)
    if centroid is None:
        return True, 1.0

    prompt_vec = get_embedding(prompt)
    if prompt_vec is None:
        return True, 1.0

    if hasattr(prompt_vec, 'tolist'):
        prompt_vec = prompt_vec.tolist()

    score = cosine_similarity(prompt_vec, centroid)
    return score >= DOMAIN_THRESHOLD, score