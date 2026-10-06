# core/risk.py
import re
import json
import os
from core.semantic_judge import semantic_judge
from core.embeddings import get_embedding, cosine_similarity
from core.cache import lookup_cache, save_cache_entry, security_cache_scope
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
from core.fusion import fused_threat_score
from core.assess_result import AssessResult
import time

logger = get_logger(__name__)

# --- 1. LOAD POLICIES ---
def load_policies():
    """
    Loads semantic threat anchors from POLICY_FILE.

    Threat anchors are grouped by attack class (`threat_anchor_classes`) so the
    threat taxonomy is explicit and per-class metrics are possible. A flat
    `threat_anchors` list is still accepted for backwards compatibility.

    Returns (threat_anchors_flat, threat_anchor_classes).
    """
    if not os.path.exists(POLICY_FILE):
        logger.warning(f"{POLICY_FILE} not found. Semantic threat detection disabled.")
        return [], {}

    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("threat_anchor_classes", {})

    if classes:
        flat = [a for anchors in classes.values() for a in anchors]
    else:
        # Legacy flat format.
        flat = data.get("threat_anchors", [])
        classes = {"unclassified": flat} if flat else {}

    return flat, classes


THREAT_ANCHORS, THREAT_ANCHOR_CLASSES = load_policies()

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
from core.policy_loader import (  # noqa: E402
    get_jailbreak_patterns, get_instruction_override_patterns, get_hard_ban_keywords,
)

JAILBREAK_PATTERNS = get_jailbreak_patterns()
INSTRUCTION_OVERRIDE_PATTERNS = get_instruction_override_patterns()
HARD_BAN_KEYWORDS = get_hard_ban_keywords()

def check_symbolic_violations(prompt: str) -> str | None:
    # FAIL-CLOSED: If symbolic rules failed to load, block everything
    if (JAILBREAK_PATTERNS is None or INSTRUCTION_OVERRIDE_PATTERNS is None
            or HARD_BAN_KEYWORDS is None):
        return "SYMBOLIC_POLICY_MISSING"
    prompt_lower = prompt.lower()
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, prompt_lower):
            return "JAILBREAK_DETECTED"
    for pattern in INSTRUCTION_OVERRIDE_PATTERNS:
        if re.search(pattern, prompt_lower):
            return "INSTRUCTION_OVERRIDE_DETECTED"
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
    return max_score > EDUCATIONAL_THRESHOLD

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

def _fast_path_signals(prompt_vec) -> dict:
    """The two cheap, vector-only signals, computed once and reused whether
    or not they end up deciding anything — collect_semantic_signals accepts
    these back so Stage 2 never repeats the same dot products."""
    return {
        "meta_intent_score": float(check_meta_intent(prompt_vec)),
        "threat_score": float(threat_store.get_max_similarity(prompt_vec)),
    }


def _fast_path_decision(fast: dict):
    """Return a decisive cheap-signal risk/source pair or None.

    Uses the configured deep-path thresholds; ambiguous signals continue to
    full collection/fusion rather than fabricating a verdict."""
    if fast["meta_intent_score"] >= META_INTENT_THRESHOLD:
        logger.warning(f"🚫 META-INTENT DETECTED (score: {fast['meta_intent_score']:.3f}) — fast path")
        return "HIGH", "fast_path_meta_intent"

    if fast["threat_score"] >= SEMANTIC_THRESHOLD_HIGH:
        logger.warning(f"🚫 ANCHOR THREAT CRITICAL (score: {fast['threat_score']:.3f}) — fast path")
        return "HIGH", "fast_path_anchor_critical"

    return None


# ============================================================================
# STAGE 2: PARALLEL SIGNAL COLLECTION
# ============================================================================

def collect_semantic_signals(prompt: str, prompt_vec, fast: dict | None = None) -> dict:
    """Stage 2: Collects all semantic signals without making any blocking decisions.
        Returns a dict of raw signal values for downstream fusion."""
    # FIX: Initialise accumulator dict and first timing checkpoint here.
    details: dict = {}

    if fast is not None:
        details["meta_intent_ms"] = 0.0  # already paid in Stage 1.5, not here
        details["meta_intent_score"] = fast["meta_intent_score"]
        details["faiss_threat_search_ms"] = 0.0
        details["threat_score"] = fast["threat_score"]
    else:
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
    details["is_educational"] = check_educational_context(prompt_vec)

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

    t0 = time.perf_counter()
    fusion = fused_threat_score(prompt, anchor_score=details["threat_score"])
    t1 = time.perf_counter()
    details["fusion_ms"] = round((t1 - t0) * 1000, 2)
    details["fusion_available"] = fusion["available"]
    details["fusion_score"] = fusion["score"]
    details["fusion_threshold_high"] = fusion["threshold_high"]
    details["fusion_threshold_medium"] = fusion["threshold_medium"]
    details["fusion_detail"] = fusion["detail"]
    details["fusion_detector_scores"] = fusion["detector_scores"]
    details["fusion_triggering_class"] = fusion.get("triggering_class")
    details["fusion_class_scores"] = fusion.get("class_scores", {})
    if not fusion["available"]:
        logger.warning(
            f"Fusion unavailable ({fusion['detail']}) - falling back to the "
            f"anchors-only decision path for this request."
        )

    return details

# ============================================================================
# STAGE 3: DETERMINISTIC FUSION
# ============================================================================

def classify_topicality(signals: dict) -> str:
    """Implementation of classify topicality."""
    if signals.get("domain_aligned") is None:
        return "UNKNOWN"
    return "IN_DOMAIN" if signals["domain_aligned"] else "OUT_OF_DOMAIN"


def _in_upper_ambiguous_band(score: float, threshold_medium: float, threshold_high: float) -> bool:
    """Distinguish the upper ambiguous band from the lower judge zone.

    A nonzero threat score alone is insufficient because the caller already
    enters this branch after the medium threshold."""
    band = threshold_high - threshold_medium
    if band <= 0:
        # Degenerate config (medium >= high) — fail toward the more
        # conservative reading rather than divide by a non-positive band.
        return True
    return score >= threshold_medium + band / 2.0


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

    # 3) Primary threat signal — fused multi-detector score when available,
    #    else the original anchors-only cosine similarity. Both branches use
    #    the SAME decision structure (HIGH cutoff, MEDIUM/judge zone, clean
    #    pass); only the signal and its calibrated thresholds differ. Sources
    #    are labelled distinctly ("fusion_*" vs the legacy "vector_*" /
    #    "clean_pass" names) so audit logs always show which decision system
    #    actually fired for a given request.
    if signals.get("fusion_available"):
        score = signals["fusion_score"]
        threshold_high = signals["fusion_threshold_high"]
        threshold_medium = signals["fusion_threshold_medium"]
        logger.debug(f"🔍 Fusion Score: {score:.3f}")

        if score >= threshold_high:
            return "HIGH", "fusion_threat_critical", False, topicality

        if score >= threshold_medium:
            if signals["is_educational"]:
                logger.info("🛡️ SAFE HARBOR: Threat detected but Context is Educational.")
                return "MEDIUM", "fusion_educational_safe_harbor", False, topicality
            return "MEDIUM", "fusion_judge_pending", True, topicality

        return "LOW", "fusion_clean_pass", False, topicality

    # Fallback: fusion unavailable for this request (see collect_semantic_signals
    # for why — a warning is already logged there). This is exactly the
    # anchors-only decision path this project shipped and validated before
    # fusion existed, kept verbatim as a safety net rather than removed.
    logger.debug(f"🔍 Threat Score (anchors-only fallback): {signals['threat_score']:.3f}")

    if signals["threat_score"] >= SEMANTIC_THRESHOLD_HIGH:
        return "HIGH", "vector_threat_critical", False, topicality

    if signals["threat_score"] >= SEMANTIC_THRESHOLD_MEDIUM:
        if signals["is_educational"]:
            logger.info("🛡️ SAFE HARBOR: Threat detected but Context is Educational.")
            return "MEDIUM", "educational_safe_harbor", False, topicality
        else:
            return "MEDIUM", "judge_pending", True, topicality

    return "LOW", "clean_pass", False, topicality

# ============================================================================
# STAGE 4: JUDGE ARBITRATION
# ============================================================================

def llama_guard_arbitration(prompt: str, threat_present: bool = False):
    """Arbitrate with the Llama Guard safety classifier when available.

    This is a judge path, not an always-on fusion feature. Availability/failure
    remain explicit; safety arbitration must not manufacture a permissive
    outcome from a missing model."""
    from core.detectors import get_detector
    from core.circuit_breaker import llama_guard_breaker

    if llama_guard_breaker.is_open():
        return None

    detector = get_detector("llama_guard_3_1b")
    ok, detail = detector.available()
    if not ok:
        logger.warning(
            f"Llama Guard arbitration unavailable ({detail}); "
            f"falling back to the Ollama judge."
        )
        return None

    try:
        result = detector.classify(prompt)
        llama_guard_breaker.record_success()
    except Exception as e:
        llama_guard_breaker.record_failure()
        logger.warning(
            f"Llama Guard arbitration raised {type(e).__name__}: {e}; "
            f"falling back to the Ollama judge."
        )
        return None

    logger.info(f"⚖️  Llama Guard verdict: {result['verdict']} {result['categories']}")

    if result["verdict"] == "unsafe":
        return "HIGH", "llama_guard_arbitration"

    # verdict == "safe": same anti-escape restriction as the Ollama judge —
    # threat_present means the score is in the UPPER half of the ambiguous
    # band specifically (see _in_upper_ambiguous_band), not merely that this
    # function was reached at all. Only there does "safe" cap at MEDIUM
    # rather than fully clearing the request.
    if threat_present or not settings.JUDGE_MAY_CLEAR_TO_LOW:
        logger.warning(
            "⚠️ JUDGE RESTRICTION: Llama Guard SAFE verdict overridden to "
            "MEDIUM (deterministic layer flagged this prompt)."
        )
        # Two distinct sources, deliberately, even though both cap at MEDIUM:
        # the audit record should distinguish "the score was high enough that
        # we would have restricted regardless" from "policy denied the arbiter
        # clear-authority". They answer different questions after an incident.
        source = ("llama_guard_override_restricted" if threat_present
                  else "llama_guard_override_capped")
        return "MEDIUM", source
    return "LOW", "llama_guard_override"


def llama_guard_async_confirmation(prompt: str, prompt_vec, fast_risk: str, fast_source: str) -> None:
    """Run background confirmation after the caller response.

    Uses the captured input/security scope and can escalate retained cache
    state. It is not synchronous clearance or permission to release output."""
    try:
        result = llama_guard_arbitration(prompt, threat_present=(fast_risk in ("MEDIUM", "HIGH")))
    except Exception as e:
        logger.warning(f"Llama Guard async confirmation raised {type(e).__name__}: {e}; fast-path verdict stands unverified.")
        return

    if result is None:
        logger.info("Llama Guard async confirmation unavailable; fast-path verdict stands unverified.")
        return

    lg_risk, lg_source = result
    if lg_risk == "HIGH" and fast_risk != "HIGH":
        logger.error(
            f"🚨 ASYNC ESCALATION: fast path served {fast_risk} ({fast_source}), "
            f"Llama Guard confirmation says HIGH ({lg_source}). Upgrading the "
            f"cache entry — this prompt, and near-duplicates via the fuzzy "
            f"cache tier, will be blocked from now on."
        )
        try:
            save_cache_entry(prompt, prompt_vec, "HIGH", 1.0, source="llama_guard_async_escalation")
        except Exception as e:
            logger.error(f"Failed to persist async escalation to cache: {type(e).__name__}: {e}")
    else:
        logger.info(
            f"Llama Guard async confirmation: fast={fast_risk} ({fast_source}), "
            f"llama_guard={lg_risk} ({lg_source}) — agrees or is more lenient; "
            f"no cache change (a served decision is never downgraded)."
        )


def judge_arbitration(prompt: str, threat_present: bool = False) -> tuple:
    """
    Stage 4: Invokes the semantic judge for ambiguous cases.
    Returns (final_risk, source).
    Fail-closed: any unrecognized verdict results in HIGH risk.

    If threat_present is True, the judge is NOT allowed to downgrade to LOW —
    a SAFE verdict is restricted to MEDIUM to prevent adversarial escape.
    `threat_present` means the score is in the UPPER half of the ambiguous
    band (close to threshold_high), not merely "ambiguous" — see
    _in_upper_ambiguous_band's docstring for why that distinction is the
    entire point. Every prompt reaching this function is already ambiguous by
    definition; treating that as the trigger made the restriction apply
    unconditionally to every call, regardless of judge quality.

    This is the FALLBACK arbiter, tried when llama_guard_arbitration is
    unavailable — see assess_risk's Stage 4 for the preference order.
    """
    logger.info("⚖️  Invoking Semantic Judge...")
    judge_verdict = semantic_judge(prompt)

    if judge_verdict == "DANGEROUS":
        return "HIGH", "semantic_judge"
    elif judge_verdict == "SAFE":
        if threat_present or not settings.JUDGE_MAY_CLEAR_TO_LOW:
            logger.warning("⚠️ JUDGE RESTRICTION: SAFE verdict overridden to MEDIUM (deterministic layer flagged this prompt).")
            source = ("semantic_judge_override_restricted" if threat_present
                      else "semantic_judge_override_capped")
            return "MEDIUM", source
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

def assess_risk(prompt: str, background_scheduler=None, raw_prompt: str | None = None, tenant: str = "default") -> tuple:
    from contextvars import copy_context
    with security_cache_scope(raw_prompt if raw_prompt is not None else prompt, tenant):
        scheduler = None
        if background_scheduler is not None:
            def scheduler(function, *args):
                context = copy_context()
                background_scheduler(context.run, function, *args)
        return _assess_risk(prompt, scheduler, raw_prompt)


def _assess_risk(prompt: str, background_scheduler=None, raw_prompt: str | None = None) -> tuple:
    """Assess current symbolic veto, scoped cache, detector signals and fusion.

    Threat risk and topicality remain separate. Required judge paths fail
    closed. Local classifiers may use raw_prompt; downstream judges/cache use
    privacy-processed content within the captured security identity."""
    text_for_detectors = raw_prompt if raw_prompt is not None else prompt

    from core import metrics as _metrics
    # ---- STAGE 1: HARD BAN (SYMBOLIC VETO) ----
    # Includes normalize_prompt, which hard_ban_triggered runs internally as
    # its own first step -- not split into a separate stage, since it is one
    # small preprocessing call with nothing else on that path to time against.
    _t0 = time.perf_counter()
    triggered, detail = hard_ban_triggered(text_for_detectors)
    _metrics.stage_duration_seconds.labels(stage="symbolic").observe(
        time.perf_counter() - _t0
    )
    if triggered:
        # HARD RULE: Educational context NEVER overrides Symbolic Violations
        return AssessResult.symbolic(detail).as_return()

    # ---- INIT: Ensure FAISS indexes are populated (lazy, once only) ----
    _ensure_faiss_initialized()

    _t0 = time.perf_counter()
    prompt_vec = get_embedding(prompt)
    cached_risk, cached_score = lookup_cache(prompt, prompt_vec)
    from core import metrics as _metrics
    _metrics.stage_duration_seconds.labels(stage="cache_lookup").observe(
        time.perf_counter() - _t0
    )
    if cached_risk:

        # SAFETY: Never downgrade a HIGH-risk cached decision
        if cached_risk == "HIGH":
            logger.info("⚡ CACHE HIT (LOCKED HIGH) — cached HIGH cannot be downgraded.")
            return AssessResult.cache_hit("HIGH", cached_score, locked_high=True).as_return()
        logger.info(f"⚡ CACHE HIT! Risk: {cached_risk}")
        return AssessResult.cache_hit(cached_risk, cached_score).as_return()

    # ---- STAGE 1.5: FAST PATH (cheap, escalate-only cascade) ----
    # See _fast_path_decision's docstring for why this is safe: it can only
    # ESCALATE to HIGH, never ALLOW, so a prompt that clears it is in
    # EXACTLY the same position as before this stage existed — it still
    # goes through the full deep path below.
    fast = _fast_path_signals(prompt_vec)
    fast_decision = _fast_path_decision(fast)
    if fast_decision is not None:
        risk, source = fast_decision
        save_cache_entry(prompt, prompt_vec, risk, 1.0, source=source)
        return AssessResult.fast_path(risk, source, fast["meta_intent_score"]).as_return()

    # ---- STAGE 2: COLLECT SEMANTIC SIGNALS ----
    signals = collect_semantic_signals(text_for_detectors, prompt_vec, fast=fast)

    # ---- STAGE 3: DETERMINISTIC FUSION ----
    risk, source, judge_required, topicality = fuse_signals(signals, text_for_detectors)

    judge_invoked = False

    # ---- STAGE 4: JUDGE ARBITRATION (if required) ----
    if judge_required:
        judge_invoked = True
        # See _in_upper_ambiguous_band's docstring: judge_required is ONLY
        # ever True when the score already cleared threshold_medium (that is
        # what "ambiguous zone" means in fuse_signals), so re-checking
        # `score >= threshold_medium` here would be tautologically True on
        # every single call — that was the actual bug. threat_present now
        # asks a real question: is this score in the upper half of the
        # ambiguous band, close to threshold_high, where a SAFE verdict
        # should still be treated with suspicion?
        if signals.get("fusion_available"):
            threat_present = _in_upper_ambiguous_band(
                signals["fusion_score"],
                signals["fusion_threshold_medium"],
                signals["fusion_threshold_high"],
            )
        else:
            threat_present = _in_upper_ambiguous_band(
                signals["threat_score"], SEMANTIC_THRESHOLD_MEDIUM, SEMANTIC_THRESHOLD_HIGH
            )

        # Timed as one "judge" stage covering whichever branch below actually
        # runs in-request. The async path's llama_guard_async_confirmation
        # call is deliberately EXCLUDED -- it runs on background_scheduler
        # after the response is already sent, so it is not part of what a
        # caller's p95 measures.
        _t0 = time.perf_counter()
        if background_scheduler is not None:
            # ASYNC PATH (live API traffic): answer now with the fast Ollama
            # judge; verify with Llama Guard afterwards, without making the
            # caller wait for it. See llama_guard_async_confirmation for the
            # escalate-only correction this applies if the two disagree.
            risk, source = judge_arbitration(prompt, threat_present=threat_present)
            _metrics.stage_duration_seconds.labels(stage="judge").observe(
                time.perf_counter() - _t0
            )
            background_scheduler(
                llama_guard_async_confirmation, prompt, prompt_vec, risk, source
            )
        else:
            # SYNCHRONOUS PATH (evaluation/benchmarking, or any caller that
            # did not opt into async): unchanged from before — try Llama
            # Guard in-request, fall back to the Ollama judge if unavailable.
            llama_guard_result = llama_guard_arbitration(prompt, threat_present=threat_present)
            if llama_guard_result is not None:
                risk, source = llama_guard_result
            else:
                risk, source = judge_arbitration(prompt, threat_present=threat_present)
            _metrics.stage_duration_seconds.labels(stage="judge").observe(
                time.perf_counter() - _t0
            )

    # ---- STAGE 5: CACHE SAVE + RETURN ----
    # Report the score that actually drove the decision: the fused probability
    # when fusion ran, else the legacy anchors-only similarity. Reporting the
    # anchors score while a fusion score decided the outcome would make the
    # audit trail describe a different decision than the one actually made.
    if signals.get("fusion_available"):
        reported_score = signals["fusion_score"]
    else:
        reported_score = signals["threat_score"]

    save_cache_entry(prompt, prompt_vec, risk, reported_score, source=source)
    # Per-stage timings (meta_intent_ms / faiss_threat_search_ms /
    # domain_alignment_ms / fusion_ms) are measured in Stage 2 and only the
    # deep path pays them. AssessResult.deep_path makes every one a REQUIRED
    # kwarg, so a future edit that forgets one is a TypeError here rather than
    # a silently empty stage_duration_seconds histogram (STAGE_KEYS in
    # core/metrics.py — the Aug "Finding C" regression).
    return AssessResult.deep_path(
        risk, source,
        semantic_score=reported_score,
        is_educational=signals["is_educational"],
        domain_score=signals["domain_score"],
        topicality=topicality,
        judge_invoked=judge_invoked,
        centroid_score=signals["centroid_score"],
        fusion_available=signals.get("fusion_available"),
        fusion_detail=signals.get("fusion_detail"),
        fusion_detector_scores=signals.get("fusion_detector_scores"),
        fusion_triggering_class=signals.get("fusion_triggering_class"),
        fusion_class_scores=signals.get("fusion_class_scores", {}),
        anchor_threat_score=signals["threat_score"],
        meta_intent_score=signals["meta_intent_score"],
        meta_intent_ms=signals.get("meta_intent_ms"),
        faiss_threat_search_ms=signals.get("faiss_threat_search_ms"),
        domain_alignment_ms=signals.get("domain_alignment_ms"),
        fusion_ms=signals.get("fusion_ms"),
    ).as_return()
