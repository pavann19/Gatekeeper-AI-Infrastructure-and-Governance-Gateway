"""Fit a plain-JSON fusion artifact from supplied detector score rows.

Persist scaler, coefficients, intercepts and decision thresholds for runtime
use. All-data fit statistics are training sanity checks, not independent
accuracy estimates. Artifact regeneration is a reviewed decision change."""
import json
import os
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import roc_auc, threshold_at_fpr

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
ARTIFACT_FILE = os.path.join("models", "fusion_policy.json")

# Order matters: each list IS the feature order baked into its tier.
# core/fusion.py scores detectors in this exact order at inference time.
FLOOR_FEATURES = ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"]
TIER_5_FEATURES = FLOOR_FEATURES + ["deepset_injection"]
TIER_6_FEATURES = TIER_5_FEATURES + ["prompt_guard_2"]
TIER_8_FEATURES = TIER_6_FEATURES + ["german_toxicity_eistakovskii", "german_toxicity_ankekat"]
TIER_9_FEATURES = TIER_8_FEATURES + ["multilingual_head"]

# Kept for backward compatibility with anything importing this name.
LIVE_FEATURES = FLOOR_FEATURES

HIGH_FPR_BUDGET = 0.05
MEDIUM_FPR_BUDGET = 0.20

PER_CLASS_HIGH_FPR_BUDGET = 0.0224
PER_CLASS_MEDIUM_FPR_BUDGET = 0.0895


def load_suite():
    rows = []
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_scores(name):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing cached scores for '{name}' at {path}.\n"
            f"Run: python -m scripts.compare_detectors"
        )
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def fit_tier(features, rows, caches, tier_id=None):
    """
    Fits one complete tier (global policy + per-class policies) for the
    given feature list, refit on ALL usable data — see module docstring
    for why the deployed artifact is refit rather than reusing an
    out-of-fold model. Returns a self-contained tier dict; the floor
    tier's dict IS the artifact's top-level fields (tier_id=None omits the
    key so the floor tier's shape matches every artifact before tiers
    existed), an upgrade tier's dict carries its own `tier_id`.
    """
    usable = [r for r in rows if all(r["id"] in caches[f] for f in features if f != "anchors")]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"  [{tier_id or 'floor'}] dropping {dropped} rows missing a cached score")

    def feature_value(r, f):
        return r["_anchor_placeholder"] if f == "anchors" else caches[f][r["id"]]

    X = [[feature_value(r, f) for f in features] for r in usable]
    y = [r["label"] for r in usable]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_scaled, y)

    # Threshold selection operates on the PROBABILITY output, on the same data
    # the model was fit on. This is the deploy-time artifact, not an unbiased
    # accuracy estimate — the unbiased estimate is scripts.sweep_fusion_
    # variants' out-of-fold AUC, reported in the engineering assessment.
    probs = model.predict_proba(X_scaled)[:, 1]
    auc_insample = roc_auc(list(probs), y)
    threshold_high = threshold_at_fpr(list(probs), y, budget=HIGH_FPR_BUDGET)
    threshold_medium = threshold_at_fpr(list(probs), y, budget=MEDIUM_FPR_BUDGET)

    # ---- PER-CLASS POLICIES ----
    # One policy per attack class (that class positive, benign negative), so
    # each gets its own decision boundary — see module docstring for why a
    # per-class threshold needs a per-class score. Budgets are per-class
    # (PER_CLASS_*_FPR_BUDGET), calibrated so the union of all classes' FPR
    # lands on the global policy's own budget, not three times it.
    classes = sorted({r["attack_class"] for r in usable if r["label"] == 1})
    per_class = {}
    for cls in classes:
        idx = [i for i, r in enumerate(usable)
               if r["label"] == 0 or r["attack_class"] == cls]
        Xc = [X[i] for i in idx]
        yc = [1 if usable[i]["attack_class"] == cls else 0 for i in idx]
        if len(set(yc)) < 2:
            continue

        cls_scaler = StandardScaler()
        Xc_scaled = cls_scaler.fit_transform(Xc)
        cls_model = LogisticRegression(max_iter=2000, C=1.0)
        cls_model.fit(Xc_scaled, yc)

        cls_probs = cls_model.predict_proba(Xc_scaled)[:, 1]
        per_class[cls] = {
            "scaler_mean": cls_scaler.mean_.tolist(),
            "scaler_scale": cls_scaler.scale_.tolist(),
            "coefficients": cls_model.coef_[0].tolist(),
            "intercept": float(cls_model.intercept_[0]),
            "threshold_high": float(threshold_at_fpr(list(cls_probs), yc, budget=PER_CLASS_HIGH_FPR_BUDGET)),
            "threshold_medium": float(threshold_at_fpr(list(cls_probs), yc, budget=PER_CLASS_MEDIUM_FPR_BUDGET)),
            "n_positive": int(sum(yc)),
        }

    tier = {
        "feature_order": features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold_high": float(threshold_high),
        "threshold_medium": float(threshold_medium),
        "fpr_budget_high": HIGH_FPR_BUDGET,
        "fpr_budget_medium": MEDIUM_FPR_BUDGET,
        "per_class": per_class,
        "per_class_fpr_budget_high": PER_CLASS_HIGH_FPR_BUDGET,
        "per_class_fpr_budget_medium": PER_CLASS_MEDIUM_FPR_BUDGET,
        "training": {
            "n_rows": len(usable),
            "n_attack": int(sum(y)),
            "n_benign": int(len(y) - sum(y)),
            "in_sample_auc": auc_insample,
            "note": "in-sample AUC is NOT an accuracy estimate (fit and scored on "
                    "the same data) - it exists only as a load-bearing sanity "
                    "check that persistence round-trips correctly. The unbiased "
                    "estimate is scripts.sweep_fusion_variants' out-of-fold AUC "
                    "for this exact feature pool, reported in "
                    "docs/EVALUATION.md.",
        },
    }
    if tier_id is not None:
        tier["tier_id"] = tier_id

    print(f"  [{tier_id or 'floor'}] {len(usable)} rows, features={features}")
    print(f"    threshold_high={threshold_high:.4f}  threshold_medium={threshold_medium:.4f}  "
          f"in-sample AUC={auc_insample:.4f}")
    return tier


def main():
    rows_raw = load_suite()
    all_features = sorted({f for f in TIER_9_FEATURES if f != "anchors"})
    caches = {name: load_scores(name) for name in all_features}

    # "anchors" isn't a cached detector score (core/fusion.py receives it
    # from the caller — see module docstring) but every row needs SOME
    # value in that column to fit against; anchors' own score is cached by
    # scripts.compare_detectors like any other feature, so reuse it here
    # too rather than inventing a placeholder.
    anchor_cache = load_scores("anchors")
    rows = [r for r in rows_raw if r["id"] in anchor_cache]
    for r in rows:
        r["_anchor_placeholder"] = anchor_cache[r["id"]]

    floor = fit_tier(FLOOR_FEATURES, rows, caches)
    tier_5 = fit_tier(TIER_5_FEATURES, rows, caches, tier_id="five_feature")
    tier_6 = fit_tier(TIER_6_FEATURES, rows, caches, tier_id="six_feature")
    tier_8 = fit_tier(TIER_8_FEATURES, rows, caches, tier_id="eight_feature")
    # multilingual_head's cached score here is 5-fold OUT-OF-FOLD (from
    # scripts.build_multilingual_feature), not the deployed all-data-refit
    # artifact's own prediction — deliberately: fitting the fusion against
    # in-sample-perfect scores from a head that has seen every row would
    # teach the fusion to over-trust this one feature. OOF is the honest
    # stand-in for "what this feature looks like on a prompt it hasn't
    # seen", which is the only case that matters at inference time.
    tier_9 = fit_tier(TIER_9_FEATURES, rows, caches, tier_id="nine_feature")

    artifact = {
        # v3 adds `upgrade_tiers`. Every top-level field is exactly the
        # floor tier's own — an artifact with no upgrade_tiers key (or a
        # core/fusion.py that predates tiers) behaves identically to
        # every artifact before this version.
        "version": 3,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **floor,
        "upgrade_tiers": [tier_9, tier_8, tier_6, tier_5],  # richest first
    }

    os.makedirs(os.path.dirname(ARTIFACT_FILE), exist_ok=True)
    with open(ARTIFACT_FILE, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact -> {ARTIFACT_FILE}")
    print(f"Floor: {FLOOR_FEATURES}")
    print("Upgrade tiers (richest first): nine_feature, eight_feature, six_feature, five_feature")


if __name__ == "__main__":
    main()
