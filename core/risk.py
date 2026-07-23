# core/risk.py
import re
import json
import os
from core.semantic_judge import semantic_judge
from core.embeddings import get_embedding, cosine_similarity
from core.cache import lookup_cache, save_cache_entry
from core.updates import check_dynamic_threats, check_dynamic_safe_harbors
from core.domain_classifier import is_domain_aligned
from core.config import (
    SEMANTIC_THRESHOLD_HIGH,
    SEMANTIC_THRESHOLD_MEDIUM,
    EDUCATIONAL_THRESHOLD,
    META_INTENT_THRESHOLD,
    POLICY_FILE,
    settings,
)
from core.normalizer import normalize_prompt
from core.threat_centroid import compute_centroid_similarity
from core.vector_store import threat_store, educational_store
from core.logger import get_logger
import time

logger = get_logger(__name__)

# --- 1. LOAD POLICIES ---
def load_policies():
    """
    Loads safe and threat anchors from POLICY_FILE.

    Threat anchors are grouped by attack class (`threat_anchor_classes`) so the
    threat taxonomy is explicit and per-class metrics are possible. A flat
    `threat_anchors` list is still accepted for backwards compatibility.

    Returns (safe_anchors, threat_anchors_flat, threat_anchor_classes).
    """
    if not os.path.exists(POLICY_FILE):
        logger.warning(f"{POLICY_FILE} not found. Semantic threat detection disabled.")
        return [], [], {}

    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    safe = data.get("safe_anchors", [])
    classes = data.get("threat_anchor_classes", {})

    if classes:
        flat = [a for anchors in classes.values() for a in anchors]
    else:
        # Legacy flat format.
        flat = data.get("threat_anchors", [])
        classes = {"unclassified": flat} if flat else {}

    return safe, flat, classes


EDUCATIONAL_ANCHORS, THREAT_ANCHORS, THREAT_ANCHOR_CLASSES = load_policies()

# EXPANDED ANCHORS for Educational Intent (Safe Harbor)
EDUCATIONAL_CONTEXT_ANCHORS = [
    "I am researching for a university cybersecurity course",
    "This is a captured flag (CTF) exercise in a virtual environment",
    "Explain the theoretical concept for exam preparation",
    "I need to understand the vulnerability to patch it",
    "academic research on malware behavior in sandboxes",
    "simulating attack vectors for defensive analysis"
]

# FAISS indexes are populated lazily on first call to assess_risk()
# to avoid sentence_transformers import at module load time (breaks CI mocking).
_faiss_initialized = False

def _ensure_faiss_initialized():
    """Populates FAISS indexes once on first use."""
    global _faiss_initialized
    if not _faiss_initialized:
        counts = {k: len(v) for k, v in THREAT_ANCHOR_CLASSES.items()}
        logger.info(f"Threat anchors by class: {counts} (total {len(THREAT_ANCHORS)})")
        threat_store.add_texts(THREAT_ANCHORS)
        educational_store.add_texts(EDUCATIONAL_CONTEXT_ANCHORS)
        _faiss_initialized = True

# --- 2. SYMBOLIC RULES (from centralized policy loader) ---
from core.policy_loader import get_jailbreak_patterns, get_hard_ban_keywords

JAILBREAK_PATTERNS = get_jailbreak_patterns()
HARD_BAN_KEYWORDS = get_hard_ban_keywords()

def check_symbolic_violations(prompt: str) -> str:
    # FAIL-CLOSED: If symbolic rules failed to load, block everything
    if JAILBREAK_PATTERNS is None or HARD_BAN_KEYWORDS is None:
        return "SYMBOLIC_POLICY_MISSING"
    prompt_lower = prompt.lower()
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, prompt_lower):
            return "JAILBREAK_DETECTED"
    for keyword in HARD_BAN_KEYWORDS:
        if keyword in prompt_lower:
            return "HARD_BAN_DETECTED"
    return None

# --- 2.5. SEMANTIC META-INTENT DETECTION ---
META_INTENT_FILE = "policies/meta_intent_anchors.json"

# META_INTENT_VECTORS is lazy-initialized on first use (same reason as FAISS above)
_meta_intent_vectors_cache = None
_meta_intent_initialized = False

def _load_meta_intent_vectors():
    """Loads meta-intent anchors and precomputes their embeddings.
    Returns list of (text, vector) tuples, or empty list if file missing."""
    if not os.path.exists(META_INTENT_FILE):
        logger.warning(f"{META_INTENT_FILE} not found. Meta-intent detection disabled.")
        return []
    try:
        with open(META_INTENT_FILE, "r") as f:
            data = json.load(f)
            intents = data.get("meta_attack_intents", [])
            vectors = []
            for intent in intents:
                vec = get_embedding(intent)
                if vec is not None:
                    vectors.append((intent, vec))
            logger.info(f"Loaded {len(vectors)} meta-intent anchors.")
            return vectors
    except Exception as e:
        logger.warning(f"Failed to load meta-intent anchors: {e}")
        return []

def _get_meta_intent_vectors():
    """Returns meta-intent vectors, computing them once on first call."""
    global _meta_intent_vectors_cache, _meta_intent_initialized
    if not _meta_intent_initialized:
        _meta_intent_vectors_cache = _load_meta_intent_vectors()
        _meta_intent_initialized = True
    return _meta_intent_vectors_cache

def check_meta_intent(prompt_vec) -> float:
    """Computes max similarity between prompt and meta-intent anchors.
    Returns the max similarity score, or 0.0 if no anchors loaded."""
    vectors = _get_meta_intent_vectors()
    if not vectors:
        return 0.0
    max_score = 0.0
    for intent_text, intent_vec in vectors:
        score = cosine_similarity(prompt_vec, intent_vec)
        if score > max_score:
            max_score = score
    return max_score



def check_educational_context(prompt_vec) -> bool:
    """
    Detects if the user is explicitly framing the request as educational/research.
    Returns: True/False based on semantic proximity to educational anchors.
    """
    # Strict threshold to prevent easy bypassing
    max_score = educational_store.get_max_similarity(prompt_vec)
    
    # Check Dynamic Safe Harbors (from Feeds)
    dyn_score = check_dynamic_safe_harbors(prompt_vec)
    
    return max(max_score, dyn_score) > EDUCATIONAL_THRESHOLD

# ============================================================================
# STAGE 1: HARD BAN (Symbolic Veto)
# ============================================================================

def hard_ban_triggered(prompt: str) -> tuple:
    """
    Stage 1: Deterministic symbolic detection.
    Normalizes prompt before checking to defeat obfuscation.
    Returns (triggered: bool, detail: str or None)
    """
    normalized = normalize_prompt(prompt)
    violation = check_symbolic_violations(normalized)
    if violation:
        logger.warning(f"🚫 SYMBOLIC BLOCK: {violation}")
        return True, violation
    return False, None

# ============================================================================
# STAGE 2: PARALLEL SIGNAL COLLECTION
# ============================================================================

def collect_semantic_signals(prompt: str, prompt_vec) -> dict:
    """
    Stage 2: Collects all semantic signals without making any blocking decisions.
    Returns a dict of raw signal values for downstream fusion.

    BUG FIX (F1): Previously, `details` and the first `t0` were used before
    being assigned, causing a NameError on every prompt that survived Stage 0
    (cache hit) and Stage 1 (hard ban).  This made the G3 pipeline completely
    non-functional in the committed snapshot and invalidated the published
    evaluation results (ASR 0%, FPR ~10%).  Both variables are now initialised
    at the start of the function before first use.
    """
    # FIX: Initialise accumulator dict and first timing checkpoint here.
    details: dict = {}
    t0 = time.perf_counter()

    # 1) Meta-intent similarity
    meta_intent_score = check_meta_intent(prompt_vec)
    t1 = time.perf_counter()
    details["meta_intent_ms"] = round((t1 - t0) * 1000, 2)
    details["meta_intent_score"] = float(meta_intent_score)

    t0 = time.perf_counter()
    # 2) Vector Threat Scan using FAISS
    threat_score = threat_store.get_max_similarity(prompt_vec)
    t1 = time.perf_counter()
    details["faiss_threat_search_ms"] = round((t1 - t0) * 1000, 2)

    details["threat_score"] = float(threat_score)
    details["dynamic_threat_score"] = float(check_dynamic_threats(prompt_vec))
    details["is_educational"] = check_dynamic_safe_harbors(prompt_vec)

    # 3) Check Domain Alignment (topicality — NOT a safety signal).
    #    Skipped entirely when the guardrail is off, since the embedding
    #    comparison is pure cost if nothing consumes the result.
    if settings.DOMAIN_GUARDRAIL_MODE == "off":
        details["domain_aligned"] = None
        details["domain_score"] = None
        details["domain_alignment_ms"] = 0.0
    else:
        t0 = time.perf_counter()
        domain_aligned, domain_score = is_domain_aligned(prompt)
        t1 = time.perf_counter()
        details["domain_alignment_ms"] = round((t1 - t0) * 1000, 2)
        details["domain_aligned"] = domain_aligned
        details["domain_score"] = domain_score

    details["centroid_score"] = compute_centroid_similarity(prompt_vec)

    return details

# ============================================================================
# STAGE 3: DETERMINISTIC FUSION
# ============================================================================

def classify_topicality(signals: dict) -> str:
    """
    Determines whether the prompt falls inside the deployment's configured
    subject domain.  This is a SCOPING judgement, not a SAFETY judgement:
    "is this prompt about my product?" is a different question from
    "is this prompt an attack?".  Conflating the two is what produced the
    ~98% false-positive rate in the pre-fix benchmark, because every benign
    off-topic prompt was scored as a malice prediction.

    Returns one of: IN_DOMAIN, OUT_OF_DOMAIN, UNKNOWN.
    """
    if signals.get("domain_aligned") is None:
        return "UNKNOWN"
    return "IN_DOMAIN" if signals["domain_aligned"] else "OUT_OF_DOMAIN"


def fuse_signals(signals: dict, prompt: str) -> tuple:
    """
    Stage 3: Deterministic decision fusion from collected signals.
    Returns (final_risk, source, judge_required, topicality).
    Does NOT invoke the judge — only flags when it is needed.

    `topicality` is reported independently of `final_risk`.  It only feeds
    into the risk decision when DOMAIN_GUARDRAIL_MODE == "enforcing", which
    is an opt-in posture for single-purpose deployments (e.g. a banking
    assistant that should refuse to discuss anything else).  The default
    ("off") means the domain corpus shipped in this repo never constrains a
    third-party deployment.
    """
    topicality = classify_topicality(signals)

    # 1) Meta-intent veto
    if signals["meta_intent_score"] >= META_INTENT_THRESHOLD:
        logger.warning(f"🚫 META-INTENT DETECTED (score: {signals['meta_intent_score']:.3f})")
        return "HIGH", "semantic_meta_intent", False, topicality

    # 2) Domain guardrail — enforcing mode only.
    if settings.DOMAIN_GUARDRAIL_MODE == "enforcing" and topicality == "OUT_OF_DOMAIN":
        logger.warning(f"⚠️ OFF-TOPIC BLOCKED (domain_score: {signals['domain_score']:.3f})")
        return "MEDIUM", "domain_guardrail", False, topicality

    if topicality == "OUT_OF_DOMAIN":
        # Advisory mode: surfaced to the caller, does not alter the risk level.
        logger.info(f"ℹ️ OFF-TOPIC (advisory, domain_score: {signals['domain_score']:.3f})")

    # 3) Vector threat scan
    logger.debug(f"🔍 Threat Score: {signals['threat_score']:.3f}")

    if signals["threat_score"] >= SEMANTIC_THRESHOLD_HIGH:
        return "HIGH", "vector_threat_critical", False, topicality

    # 4) Ambiguous zone — educational safe harbor or judge required
    if signals["threat_score"] >= SEMANTIC_THRESHOLD_MEDIUM:
        if signals["is_educational"]:
            logger.info("🛡️ SAFE HARBOR: Threat detected but Context is Educational.")
            return "MEDIUM", "educational_safe_harbor", False, topicality
        else:
            return "MEDIUM", "judge_pending", True, topicality

    # 5) Clean pass
    return "LOW", "clean_pass", False, topicality

# ============================================================================
# STAGE 4: JUDGE ARBITRATION
# ============================================================================

def judge_arbitration(prompt: str, threat_present: bool = False) -> tuple:
    """
    Stage 4: Invokes the semantic judge for ambiguous cases.
    Returns (final_risk, source).
    Fail-closed: any unrecognized verdict results in HIGH risk.

    If threat_present is True, the judge is NOT allowed to downgrade to LOW.
    A SAFE verdict is restricted to MEDIUM to prevent adversarial escape.
    """
    logger.info("⚖️  Invoking Semantic Judge...")
    judge_verdict = semantic_judge(prompt)

    if judge_verdict == "DANGEROUS":
        return "HIGH", "semantic_judge"
    elif judge_verdict == "SAFE":
        if threat_present:
            logger.warning("⚠️ JUDGE RESTRICTION: SAFE verdict overridden to MEDIUM (threat signal present).")
            return "MEDIUM", "semantic_judge_override_restricted"
        return "LOW", "semantic_judge_override"
    elif judge_verdict == "AMBIGUOUS":
        return "MEDIUM", "semantic_judge_ambiguous"
    else:
        # FAIL-CLOSED: JUDGE_OFFLINE, JUDGE_ERROR, or any unrecognized verdict
        logger.error(f"🚨 JUDGE FAILURE (verdict: {judge_verdict}) — Failing closed to HIGH.")
        return "HIGH", "judge_failure_fail_closed"

# ============================================================================
# POLICY ARBITER — STAGED ORCHESTRATOR
# ============================================================================

def assess_risk(prompt: str) -> tuple:
    """
    Staged governance pipeline:
        Stage 0: Cache lookup
        Stage 1: Hard ban (symbolic veto)
        Stage 2: Parallel signal collection
        Stage 3: Deterministic fusion
        Stage 4: Judge arbitration (only if needed)
        Stage 5: Cache save + return
    """

    # ---- INIT: Ensure FAISS indexes are populated (lazy, once only) ----
    _ensure_faiss_initialized()

    # ---- STAGE 0: CACHE CHECK ----
    prompt_vec = get_embedding(prompt)
    cached_risk, cached_score = lookup_cache(prompt_vec)
    if cached_risk:

        # SAFETY: Never downgrade a HIGH-risk cached decision
        if cached_risk == "HIGH":
            logger.info(f"⚡ CACHE HIT (LOCKED HIGH) — cached HIGH cannot be downgraded.")
            return "HIGH", {"semantic_score": cached_score, "source": "cache_locked_high",
                            "educational_context": False, "domain_score": None,
                            "topicality": "UNKNOWN",
                            "symbolic_triggered": False, "judge_invoked": False,
                            "dynamic_threat_score": None, "meta_intent_score": None}
        logger.info(f"⚡ CACHE HIT! Risk: {cached_risk}")
        return cached_risk, {"semantic_score": cached_score, "source": "cache",
                             "educational_context": False, "domain_score": None,
                             "topicality": "UNKNOWN",
                             "symbolic_triggered": False, "judge_invoked": False,
                             "dynamic_threat_score": None, "meta_intent_score": None}

    # ---- STAGE 1: HARD BAN (SYMBOLIC VETO) ----
    triggered, detail = hard_ban_triggered(prompt)
    if triggered:
        # HARD RULE: Educational context NEVER overrides Symbolic Violations
        save_cache_entry(prompt, prompt_vec, "HIGH", 1.0, source="symbolic_rule")
        return "HIGH", {"source": "symbolic_rule", "detail": detail, "semantic_score": 1.0,
                        "educational_context": False, "domain_score": None,
                        "topicality": "UNKNOWN",
                        "symbolic_triggered": True, "judge_invoked": False,
                        "dynamic_threat_score": None, "meta_intent_score": None}

    # ---- STAGE 2: COLLECT SEMANTIC SIGNALS ----
    signals = collect_semantic_signals(prompt, prompt_vec)

    # ---- STAGE 3: DETERMINISTIC FUSION ----
    risk, source, judge_required, topicality = fuse_signals(signals, prompt)

    judge_invoked = False

    # ---- STAGE 4: JUDGE ARBITRATION (if required) ----
    if judge_required:
        judge_invoked = True
        threat_present = signals["threat_score"] >= SEMANTIC_THRESHOLD_MEDIUM
        risk, source = judge_arbitration(prompt, threat_present=threat_present)

    # ---- STAGE 5: CACHE SAVE + RETURN ----
    save_cache_entry(prompt, prompt_vec, risk, signals["threat_score"], source=source)
    return risk, {"semantic_score": signals["threat_score"], "source": source,
                  "educational_context": signals["is_educational"],
                  "domain_score": signals["domain_score"],
                  "topicality": topicality,
                  "symbolic_triggered": False, "judge_invoked": judge_invoked,
                  "dynamic_threat_score": signals["dynamic_threat_score"],
                  "centroid_score": signals["centroid_score"],
                  "meta_intent_score": signals["meta_intent_score"]}