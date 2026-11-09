"""Learned threat-score fusion from plain JSON policy artifacts.

Each artifact tier declares its own feature set, scaler, coefficients and
thresholds. Detectors are scored once; the richest fully available tier is
selected. Missing features are not imputed as zero. If no tier is usable,
unavailability is explicit and the caller applies its documented fallback.
Per-class threat scores remain separate from domain/topicality. Fit metadata
is not independent accuracy evidence; see docs/EVALUATION.md."""
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_FILE = os.path.join("models", "fusion_policy.json")

# The FLOOR tier's live-model detectors — documentation only since upgrade
# tiers were added; the actual set scored at request time is computed from
# the loaded artifact's tiers (`_union_required_features`), not this
# constant. "anchors" is deliberately excluded here because core/risk.py
# already computes the equivalent signal (max FAISS threat-anchor
# similarity) in collect_semantic_signals — recomputing it via the
# AnchorDetector class would re-embed the prompt for no new information.
LIVE_MODEL_DETECTORS = ("protectai_injection", "madhurjindal_jailbreak", "toxic_bert")

_policy = None
_policy_error = None
_policy_loaded = False

# Shared worker pool for concurrent detector scoring, created lazily on first
# use so importing this module costs nothing when FUSION_PARALLEL is off.
_pool = None
_pool_lock = threading.Lock()

# Guards the one-time sequential model warm-up that must precede any parallel
# dispatch — see _warm_detectors for the import race this prevents.
_detectors_warmed = False
_warm_lock = threading.Lock()


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
    required = {"feature_order", "scaler_mean", "scaler_scale",
                "coefficients", "intercept", "threshold_high", "threshold_medium"}

    def _validate_tier(tier):
        """Raises on a malformed tier. Shared between the floor tier (which
        must be valid or the whole policy is unavailable) and each upgrade
        tier (which is independently droppable — see below)."""
        missing = required - set(tier)
        if missing:
            raise ValueError(f"artifact missing fields: {sorted(missing)}")
        n = len(tier["feature_order"])
        if not (len(tier["scaler_mean"]) == len(tier["scaler_scale"])
                == len(tier["coefficients"]) == n):
            raise ValueError("feature_order/scaler/coefficients length mismatch")

    try:
        with open(ARTIFACT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _validate_tier(data)

        # Upgrade tiers are validated independently: one malformed tier must
        # not take down the floor tier that IS valid — the same "one bad
        # entry doesn't sink the others" discipline core/tenancy.py's
        # TenantStore and core/auth.py's KeyStore already apply to their own
        # per-entry configuration.
        valid_upgrade_tiers = []
        for i, tier in enumerate(data.get("upgrade_tiers", [])):
            try:
                _validate_tier(tier)
                valid_upgrade_tiers.append(tier)
            except Exception as e:
                logger.error(f"Ignoring malformed upgrade_tiers[{i}] "
                            f"({tier.get('tier_id', '?')}): {type(e).__name__}: {e}")
        data["upgrade_tiers"] = valid_upgrade_tiers

        _policy = data
        tier_summary = ", ".join(
            f"{t.get('tier_id', '?')}({len(t['feature_order'])}f)" for t in valid_upgrade_tiers
        )
        logger.info(
            f"Fusion policy loaded: floor={data['feature_order']}, "
            f"threshold_high={data['threshold_high']:.4f}, "
            f"threshold_medium={data['threshold_medium']:.4f}"
            + (f", upgrade tiers available: {tier_summary}" if tier_summary else "")
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


def _apply_policy(feature_values: dict, policy=None, feature_order=None) -> float:
    """
    feature_values: {detector_name: raw_score}, must cover every name in
    the effective feature order. Returns P(attack) in [0, 1].

    `policy` defaults to the global (floor-tier) policy. Per-class policies
    (artifact v2) and upgrade-tier policies carry their own scaler and
    coefficients but no `feature_order` of their own — a per-class or
    per-tier policy is only ever scored against the feature order of the
    TIER it belongs to, so callers holding that context pass it explicitly
    via `feature_order`; omitting it falls back to the floor tier's, which
    is correct for every call site that predates upgrade tiers.
    """
    pol = policy if policy is not None else _policy
    if pol is None:
        raise ValueError("Fusion policy unavailable")
    if feature_order is None:
        if _policy is None:
            raise ValueError("Fusion feature order unavailable")
        order = _policy["feature_order"]
    else:
        order = feature_order
    z = pol["intercept"]
    for name, mean, scale, coef in zip(
        order, pol["scaler_mean"], pol["scaler_scale"], pol["coefficients"],
    ):
        raw = feature_values[name]
        standardized = (raw - mean) / scale if scale else 0.0
        z += coef * standardized
    return _sigmoid(z)


def _select_per_class_verdict(feature_values: dict, policy_tier=None):
    """
    Scores the prompt under EVERY per-class policy IN `tier` (defaults to
    the floor tier) and returns the most severe result, or None when that
    tier has no per-class section (v1 artifacts, an upgrade tier that
    didn't train one, or per-class scoring disabled).

    WHY PER-CLASS SCORES RATHER THAN PER-CLASS THRESHOLDS ON ONE SCORE: at
    inference the attack class is unknown — determining it is the job being
    done — so a per-class threshold is only meaningful against a per-class
    score. Each class therefore gets its own policy (that class positive,
    benign negative) and its own decision boundary, which is what letting
    harmful_content be more sensitive without loosening injection/jailbreak
    actually requires.

    Selection is by (tier, ratio): the most severe tier wins, ties broken by
    how far into its own band the score sits (`score / threshold_high`),
    which normalises across classes whose thresholds differ. Returning the
    TRIGGERING class's score and thresholds — rather than inventing a
    combined score — is what lets core/risk.py's existing decision cascade,
    including the ambiguous-band logic in `_in_upper_ambiguous_band`, work
    completely unchanged.
    """
    policy_tier = policy_tier if policy_tier is not None else _policy
    per_class = policy_tier.get("per_class")
    if not per_class:
        return None

    results = {}
    for cls, pol in per_class.items():
        score = _apply_policy(feature_values, pol, feature_order=policy_tier["feature_order"])
        thr_high = pol["threshold_high"]
        thr_med = pol["threshold_medium"]
        if score >= thr_high:
            severity_tier = 2
        elif score >= thr_med:
            severity_tier = 1
        else:
            severity_tier = 0
        results[cls] = {
            "score": score, "tier": severity_tier,
            "ratio": (score / thr_high) if thr_high else 0.0,
            "threshold_high": thr_high, "threshold_medium": thr_med,
        }

    winner = max(results, key=lambda c: (results[c]["tier"], results[c]["ratio"]))
    return winner, results


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
    import time as _time
    from core import metrics
    from core.detectors import get_detector

    t0 = _time.perf_counter()
    try:
        detector = get_detector(name)
        det_ok, det_detail = detector.available()
        if not det_ok:
            return name, None, f"detector '{name}' unavailable: {det_detail}"
        return name, detector.score_batch([prompt])[0], None
    except Exception as e:
        return name, None, f"detector '{name}' raised {type(e).__name__}: {e}"
    finally:
        # Timed unconditionally, including the unavailable/error returns above
        # -- an "unavailable" detector that takes 2s to say so (a hung auth
        # check, say) is exactly the kind of latency this metric exists to
        # surface, not a case to skip.
        metrics.detector_duration_seconds.labels(detector=name).observe(
            _time.perf_counter() - t0
        )


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


def _warm_detectors(names):
    """
    Forces every detector's lazy `_load()` to completion SEQUENTIALLY, in the
    calling thread, before any parallel dispatch.

    WHY THIS IS REQUIRED, not defensive padding. Detector `_load()` runs
    `from transformers import AutoModelForSequenceClassification` on first
    use. transformers 5.x resolves that through a lazy module loader, and
    three worker threads hitting it simultaneously on a COLD module race:
    the real failure observed was `ImportError: cannot import name
    'AutoModelForSequenceClassification' from 'transformers'` — from a
    package where that symbol plainly exists and imports fine on its own.
    The whole fusion then reported unavailable and fell back to anchors-only.

    Each detector's own `_load()` is lock-guarded, but those are per-detector
    locks; they cannot serialise a race inside the shared `transformers`
    module machinery that all three enter at once.

    Loading is a one-time cold-start cost, so serialising it gives up nothing:
    the repeated per-request work is inference, and that stays parallel. This
    is also why every mocked unit test passed — only a real cold start with
    real models reaches the import at all.
    """
    global _detectors_warmed
    if _detectors_warmed:
        return
    with _warm_lock:
        if _detectors_warmed:
            return
        from core.detectors import get_detector
        for name in names:
            try:
                get_detector(name).available()   # triggers _load(), one at a time
            except Exception as e:
                # Don't mark warmed — a transient failure should be retried on
                # the next request rather than cached as permanent. The real
                # per-detector error surfaces through _score_one_detector.
                logger.warning(f"Warm-up of detector '{name}' failed: {type(e).__name__}: {e}")
                return
        _detectors_warmed = True


def _all_tiers():
    """
    Every tier this policy can run at, RICHEST FIRST, floor tier always
    last — the floor is both the worst acceptable outcome and the one
    guaranteed to validate (see `_load_policy`), so it is always a safe
    final entry regardless of what upgrade tiers declare.
    """
    return list(_policy.get("upgrade_tiers", [])) + [_policy]


def _tier_required_features(tier):
    return [n for n in tier["feature_order"] if n != "anchors"]


def _union_required_features():
    """Every non-anchors detector name needed by ANY tier, order-stable and
    deduplicated — scored once per request regardless of how many tiers
    reference it, so trying a richer tier first never means re-scoring."""
    seen = []
    for tier in _all_tiers():
        for name in _tier_required_features(tier):
            if name not in seen:
                seen.append(name)
    return seen


def warm_up():
    """Loads the policy and every live detector, so the first real request does
        not pay for it."""
    ok, detail = policy_available()
    if not ok:
        return False, f"policy unavailable, nothing warmed: {detail}"

    names = _union_required_features()
    _warm_detectors(names)
    if _detectors_warmed:
        return True, f"warmed {len(names)} detector(s): {', '.join(names)}"
    return False, "one or more detectors failed to warm; will retry on request"


def _score_detectors_parallel(names, prompt):
    """
    Runs the detectors concurrently, returning results in DECLARED order.

    Thread safety: models are warmed sequentially first (see
    `_warm_detectors`), after which each detector owns a separate, fully
    loaded model object — no shared state, and inference is a forward pass
    under `torch.no_grad()` in eval mode, so there is no parameter mutation
    to race on.
    """
    _warm_detectors(names)
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

    names = _union_required_features()

    if settings.FUSION_PARALLEL and len(names) > 1:
        results = _score_detectors_parallel(names, prompt)
    else:
        results = [_score_one_detector(n, prompt) for n in names]

    feature_values = {"anchors": anchor_score}
    detector_scores = {"anchors": anchor_score}
    # Iterate in DECLARED order, not completion order, so the reported error
    # is deterministic and reproducible regardless of which detector happened
    # to finish first — otherwise the same failure would produce different
    # `detail` strings run to run, which is miserable to debug and impossible
    # to assert on in a test.
    for name, score, error in results:
        if error is not None:
            continue
        feature_values[name] = score
        detector_scores[name] = score

    # Pick the RICHEST tier whose required features all came back with a
    # real score. A tier is skipped, not fatal — only if the FLOOR tier
    # (always last in _all_tiers()) also can't be met does this fail closed.
    chosen = None
    for tier in _all_tiers():
        if all(n in feature_values for n in _tier_required_features(tier)):
            chosen = tier
            break

    if chosen is None:
        first_error = next((error for _, _, error in results if error is not None), None)
        return {"available": False, "score": None, "threshold_high": None,
                "threshold_medium": None,
                "detail": first_error or "no score computed for a required feature",
                "detector_scores": detector_scores}

    per_class_result = (
        _select_per_class_verdict(feature_values, chosen)
        if settings.FUSION_PER_CLASS else None
    )

    tier_note = "" if chosen is _policy else f", tier={chosen.get('tier_id', '?')}"

    if per_class_result is not None:
        winner, class_results = per_class_result
        w = class_results[winner]
        # `score` and the thresholds are the TRIGGERING class's, deliberately:
        # core/risk.py compares score against these same two thresholds, so
        # returning a matched triple keeps its decision cascade — and the
        # ambiguous-band logic — correct with no changes there at all.
        return {
            "available": True,
            "score": w["score"],
            "threshold_high": w["threshold_high"],
            "threshold_medium": w["threshold_medium"],
            "detail": f"fusion applied (per-class, triggered by {winner}{tier_note})",
            "detector_scores": detector_scores,
            "triggering_class": winner,
            "class_scores": {c: r["score"] for c, r in class_results.items()},
        }

    probability = _apply_policy(feature_values, chosen, feature_order=chosen["feature_order"])
    return {
        "available": True,
        "score": probability,
        "threshold_high": chosen["threshold_high"],
        "threshold_medium": chosen["threshold_medium"],
        "detail": f"fusion applied{tier_note}",
        "detector_scores": detector_scores,
        "triggering_class": None,
        "class_scores": {},
    }
