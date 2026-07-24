# core/threat_centroid.py
# Centroid-based malicious intent detection.
# Computes a single centroid vector from threat anchors on first use
# (lazy initialization — NOT at module import time, to allow CI mocking).

import json
import os


POLICY_FILE = "policies.json"

# Lazy state
_threat_centroid_cache = None
_threat_centroid_initialized = False


def load_threat_anchors():
    """Load threat anchor strings from policies.json."""
    if not os.path.exists(POLICY_FILE):
        print(f"WARNING: {POLICY_FILE} not found. Threat centroid disabled.")
        return []
    try:
        with open(POLICY_FILE, "r") as f:
            data = json.load(f)
        # Anchors were regrouped by attack class into `threat_anchor_classes`;
        # the flat `threat_anchors` key is retained only for backward
        # compatibility and is empty in the current policy file. Reading only
        # the flat key (as this did) silently loaded ZERO anchors, so the
        # centroid was undefined and centroid_score meaningless. Prefer the
        # grouped classes, flattened, matching core.risk.load_policies().
        classes = data.get("threat_anchor_classes")
        if classes:
            anchors = [a for group in classes.values() for a in group]
        else:
            anchors = data.get("threat_anchors", [])
        print(f"Loaded {len(anchors)} threat anchors for centroid computation.")
        return anchors
    except Exception as e:
        print(f"WARNING: Failed to load threat anchors: {e}")
        return []


def build_malicious_centroid(anchors):
    """
    Compute the centroid (element-wise average) of all threat anchor embeddings.
    Returns the centroid vector, or None if no valid embeddings.
    """
    from core.embeddings import get_embedding  # lazy import — avoids CI failure
    if not anchors:
        return None

    vectors = []
    for anchor in anchors:
        vec = get_embedding(anchor)
        if vec is not None:
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            vectors.append(vec)

    if not vectors:
        return None

    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            centroid[i] += vec[i]
    for i in range(dim):
        centroid[i] /= len(vectors)

    print(f"Malicious centroid built from {len(vectors)} vectors (dim={dim}).")
    return centroid


def _get_malicious_centroid():
    """Returns the malicious centroid, computing it once on first call."""
    global _threat_centroid_cache, _threat_centroid_initialized
    if not _threat_centroid_initialized:
        anchors = load_threat_anchors()
        _threat_centroid_cache = build_malicious_centroid(anchors)
        _threat_centroid_initialized = True
    return _threat_centroid_cache


def compute_centroid_similarity(prompt_vec):
    """
    Compute cosine similarity between prompt and the malicious centroid.
    Returns 0.0 if centroid is unavailable.
    """
    from core.embeddings import cosine_similarity  # lazy import
    centroid = _get_malicious_centroid()
    if centroid is None:
        return 0.0
    if prompt_vec is None:
        return 0.0
    pv = prompt_vec.tolist() if hasattr(prompt_vec, "tolist") else prompt_vec
    return cosine_similarity(pv, centroid)
