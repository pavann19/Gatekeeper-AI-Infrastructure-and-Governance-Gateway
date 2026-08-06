"""
Applies the trained multi-detector fusion policy to a live prompt.

WHY THIS FILE EXISTS
--------------------
`scripts.ensemble_analysis` proved that a learned fusion over several
specialised detectors beats any single detector, and — more importantly — that
it closes coverage gaps a single detector leaves open (adopting the best single
injection detector alone would have dropped harmful-content detection to 2%).
Until this file existed, none of that was true of the DEPLOYED gateway: the
live pipeline in `core/risk.py` only ever consulted the anchor detector. This
module is what lets the request path actually use the validated ensemble.

WHAT IT LOADS
-------------
`models/fusion_policy.json`, produced by `scripts.train_fusion_policy` — a
StandardScaler + LogisticRegression over four detectors chosen because they are
(a) not gated behind an external licence and (b) fast enough for synchronous
request handling: anchors, protectai_injection, madhurjindal_jailbreak,
toxic_bert. Prompt Guard 2 / Llama Guard measurably strengthen the offline
ensemble further, but the first requires a per-deployment Meta licence and the
second is far too slow for a live request (seconds, not milliseconds, on CPU) —
both remain valid opt-in upgrades, not part of this default path.

FAIL-CLOSED, BUT NOT TO A CRASH
--------------------------------
If any of the three transformer detectors fails to load (disk issue, first-run
download failure), this module does NOT block every request, and it does NOT
silently score the missing feature as zero — an imputed zero would understate
risk exactly when a detector is unavailable, which is the wrong direction to
fail. Instead `fused_threat_score` reports unavailability explicitly and the
caller (`core/risk.py`) falls back to the pre-fusion anchors-only decision path,
which is the behaviour this project shipped and validated before fusion existed.
A documented, tested fallback beats either a crash or a wrong number.
"""
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_FILE = os.path.join("models", "fusion_policy.json")

# Detectors this module runs directly; "anchors" is deliberately excluded here
# because core/risk.py already computes the equivalent signal (max FAISS
# threat-anchor similarity) in collect_semantic_signals — recomputing it via
# the AnchorDetector class would re-embed the prompt for no new information.
LIVE_MODEL_DETECTORS = ("protectai_injection", "madhurjindal_jailbreak", "toxic_bert")

_policy = None
_policy_error = None
_policy_loaded = False

# Shared worker pool for concurrent detector scoring, created lazily on first
# use so importing this module costs nothing when FUSION_PARALLEL is off.
_pool = None
_pool_lock = threading.Lock()


def _load_policy():
    global _policy, _policy_error, _policy_loaded
    if _policy_loaded:
        return
    _policy_loaded = True
    if not os.path.exists(ARTIFACT_FILE):
        _policy_error = (
            f"{ARTIFACT_FILE} not found. Train it with: "
            f"python -m scripts.train_fusion_policy"
        )
        logger.warning(f"Fusion policy unavailable: {_policy_error}")
        return
    try:
        with open(ARTIFACT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = {"feature_order", "scaler_mean", "scaler_scale",
                    "coefficients", "intercept", "threshold_high", "threshold_medium"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"artifact missing fields: {sorted(missing)}")
        n = len(data["feature_order"])
        if not (len(data["scaler_mean"]) == len(data["scaler_scale"])
                == len(data["coefficients"]) == n):
            raise ValueError("feature_order/scaler/coefficients length mismatch")
        _policy = data
        logger.info(
            f"Fusion policy loaded: {data['feature_order']}, "
            f"threshold_high={data['threshold_high']:.4f}, "
            f"threshold_medium={data['threshold_medium']:.4f}"
        )
    except Exception as e:
        _policy_error = f"{type(e).__name__}: {e}"
        logger.error(f"Fusion policy failed to load: {_policy_error}")


def policy_available():
    """Returns (available: bool, detail: str). Loads the artifact on first call."""
    _load_policy()
    if _policy is None:
        return False, _policy_error
    return True, f"loaded {ARTIFACT_FILE} ({len(_policy['feature_order'])} features)"


def _sigmoid(x):
    # Guard against overflow on an extreme (mis-scaled) input; logistic
    # regression's linear term should never realistically reach this range.
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _apply_policy(feature_values: dict) -> float:
    """
    feature_values: {detector_name: raw_score}, must cover every name in
    policy['feature_order']. Returns P(attack) in [0, 1].
    """
    z = _policy["intercept"]
    for name, mean, scale, coef in zip(
        _policy["feature_order"], _policy["scaler_mean"],
        _policy["scaler_scale"], _policy["coefficients"],
    ):
        raw = feature_values[name]
        standardized = (raw - mean) / scale if scale else 0.0
        z += coef * standardized
    return _sigmoid(z)


def _score_one_detector(name, prompt):
    """
    Scores one detector. Returns (name, score, error_detail) and NEVER raises
    — it runs inside a worker thread where an escaping exception would be
    swallowed by the executor and surface as an opaque future failure rather
    than the specific, actionable message the caller needs.

    Error strings are byte-identical to the ones the original sequential
    implementation produced, so audit records and existing tests that match
    on them keep working.
    """
    from core.detectors import get_detector

    try:
        detector = get_detector(name)
        det_ok, det_detail = detector.available()
        if not det_ok:
            return name, None, f"detector '{name}' unavailable: {det_detail}"
        return name, detector.score_batch([prompt])[0], None
    except Exception as e:
        return name, None, f"detector '{name}' raised {type(e).__name__}: {e}"


def _get_pool(n_workers):
    """
    A module-level thread pool, created once and reused.

    Creating a ThreadPoolExecutor per request would add thread-spawn overhead
    to every single request — on a path where the whole point is shaving
    milliseconds, that can erase the gain being chased.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(
                    max_workers=n_workers, thread_name_prefix="fusion-detector"
                )
    return _pool


def _score_detectors_parallel(names, prompt):
    """
    Runs the detectors concurrently, returning results in DECLARED order.

    Thread safety: each detector owns a separate model object, so no model
    state is shared across threads, and each detector's lazy `_load()` is
    already lock-guarded. Inference is a forward pass under `torch.no_grad()`
    with the model in eval mode — no parameter mutation, nothing to race on.
    """
    pool = _get_pool(len(names))
    futures = [pool.submit(_score_one_detector, n, prompt) for n in names]
    return [f.result() for f in futures]


def fused_threat_score(prompt: str, anchor_score: float) -> dict:
    """
    Computes the fused attack probability for one prompt.

    `anchor_score` is the max FAISS threat-anchor similarity that
    collect_semantic_signals already computed — passed in rather than
    recomputed, since running the AnchorDetector class here would re-embed the
    prompt for a value the caller already has.

    Returns a dict:
        available: bool
        score: float or None        — P(attack) if available
        threshold_high: float or None
        threshold_medium: float or None
        detail: str                 — human-readable status
        detector_scores: dict       — raw per-detector scores, for audit/debug

    On ANY detector failure this returns available=False rather than a partial
    or imputed score. See module docstring for why a missing detector must not
    silently become a zero.
    """
    ok, detail = policy_available()
    if not ok:
        return {"available": False, "score": None, "threshold_high": None,
                "threshold_medium": None, "detail": detail, "detector_scores": {}}

    names = [n for n in LIVE_MODEL_DETECTORS if n in _policy["feature_order"]]

    if settings.FUSION_PARALLEL and len(names) > 1:
        results = _score_detectors_parallel(names, prompt)
    else:
        results = [_score_one_detector(n, prompt) for n in names]

    feature_values = {"anchors": anchor_score}
    detector_scores = {"anchors": anchor_score}
    first_error = None
    # Iterate in DECLARED order, not completion order, so the reported error
    # is deterministic and reproducible regardless of which detector happened
    # to finish first — otherwise the same failure would produce different
    # `detail` strings run to run, which is miserable to debug and impossible
    # to assert on in a test.
    for name, score, error in results:
        if error is not None:
            if first_error is None:
                first_error = error
            continue
        feature_values[name] = score
        detector_scores[name] = score

    if first_error is not None:
        return {"available": False, "score": None, "threshold_high": None,
                "threshold_medium": None, "detail": first_error,
                "detector_scores": detector_scores}

    missing = [f for f in _policy["feature_order"] if f not in feature_values]
    if missing:
        return {"available": False, "score": None, "threshold_high": None,
                "threshold_medium": None,
                "detail": f"no score computed for required feature(s): {missing}",
                "detector_scores": detector_scores}

    probability = _apply_policy(feature_values)
    return {
        "available": True,
        "score": probability,
        "threshold_high": _policy["threshold_high"],
        "threshold_medium": _policy["threshold_medium"],
        "detail": "fusion applied",
        "detector_scores": detector_scores,
    }
