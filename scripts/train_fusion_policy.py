"""
Fits the deployed fusion policy and persists it as a plain-JSON artifact that
`core/fusion.py` loads at request time.

WHY THIS EXISTS
---------------
scripts.ensemble_analysis proved the thesis using OUT-OF-FOLD cross-validation
— that is the right way to answer "does fusion beat a single detector", because
scoring a model on the data it was fit on would guarantee a win and prove
nothing. But the artifact actually deployed is a different object: once the
architecture is validated, the model shipped to production is refit on ALL
available data, because held-out folds buy nothing at deploy time and only cost
data efficiency. Cross-validation validates the APPROACH; this script produces
the ARTIFACT.

FLOOR TIER: FOUR DETECTORS, ALWAYS DEPLOYABLE
-----------------------------------------------
The floor tier — the one every deployment can reach with no external
licence — is exactly the four-detector pool from the FIRST ensemble_
analysis run: anchors, protectai_injection, madhurjindal_jailbreak,
toxic_bert (AUC 0.944 [0.936, 0.951] against the best single detector's
0.909 [0.899, 0.919] — non-overlapping, validated). Llama Guard remains
excluded from every tier: it is far too slow for a live request path
(~seconds, not milliseconds, on CPU), which is an inference-speed problem
no amount of graceful degradation fixes.

UPGRADE TIERS: RICHER, OPTIONAL, GRACEFULLY DEGRADING
--------------------------------------------------------
`core/fusion.py`'s tier mechanism (added 2026-08-24) tries the RICHEST
tier whose required detectors are all available and only falls back to a
poorer tier when one is missing — never straight to the pre-fusion
anchors-only path the way a missing feature used to. That capability is
what makes adding `prompt_guard_2` (Meta-gated) no longer the all-or-
nothing bet it was when this script only produced one policy: a
deployment without the licence simply runs one tier down, not zero tiers.

Measured out-of-fold on the 13,011-row suite (`scripts.sweep_fusion_
variants`, `scripts.analyze_german_by_task` for the language split), all
deltas non-overlapping-CI decisive:

    floor (4-feature)                    pooled AUC 0.846, German-injection 0.813
    +deepset_injection (5, no licence)   pooled AUC 0.866, German-injection 0.971
    +prompt_guard_2 too (6, licence)     pooled AUC 0.908, German-injection 0.987
    +2 German toxicity detectors (8)     pooled AUC 0.919, German-offensive 0.741

`deepset_injection` carries no licence gate, so the 5-feature tier is
reachable by every deployment exactly like the floor tier is — it exists
as a distinct tier (rather than just raising the floor to 5 features)
specifically so a `deepset_injection` outage alone still degrades one
step, not to the floor, and so the floor tier itself stays the
minimal, maximally-portable baseline this project has always shipped.

The 8-feature tier (issue #3) adds `german_toxicity_eistakovskii` and
`german_toxicity_ankekat`, two German-trained toxicity classifiers with no
licence gate — closing the German OFFENSIVE-CONTENT gap the 6-feature
tier barely touched (AUC 0.597→0.741) while improving every other axis
too (pooled, German injection, English — none traded away; see
`scripts.analyze_german_by_task` and `scripts.sweep_fusion_variants` for
the full breakdown). Their `trained_on` is declared empty (UNKNOWN, not
verified clean) rather than confirmed disjoint from
`philschmid/germeval18` — neither model card names its training data
explicitly, the same honest caveat `NemoGuardJailbreakDetector` already
carries for the same reason.

PERSISTENCE FORMAT
-------------------
Plain JSON, not pickle/joblib. A pickled sklearn object ties the runtime to the
exact sklearn version and Python environment that trained it, which is a
needless deployment fragility. Logistic regression's inference is five numbers
per feature (scaler mean, scaler scale, coefficient) plus an intercept and two
thresholds — trivial to apply by hand at inference time (see core/fusion.py).

THRESHOLDS
----------
Two operating points, matching the existing HIGH/MEDIUM two-tier structure in
core/risk.py:
  - HIGH: threshold achieving <=5% FPR (this project's standard budget
    throughout §1d-1g of the engineering assessment).
  - MEDIUM: threshold achieving <=20% FPR, a more permissive cutoff. Prompts
    scoring above MEDIUM but below HIGH are routed to judge arbitration rather
    than auto-blocked, mirroring the existing SEMANTIC_THRESHOLD_MEDIUM /
    SEMANTIC_THRESHOLD_HIGH gap.

Usage:
    python -m scripts.train_fusion_policy
"""
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
# German offensive-content detectors (issue #3). Off-the-shelf, no licence
# gate, no stacking caveat -- unlike the multilingual_head experiment
# (issue #4), these are static public models like every other detector in
# this ensemble. Measured (scripts.analyze_german_by_task, scripts.
# sweep_fusion_variants): decisive, no-regression improvement over tier 6
# on every axis -- pooled AUC 0.908->0.919, German 0.729->0.791, English
# 0.941->0.950 (flat-to-better, not traded away), "other" languages within
# overlapping CIs (not a regression). German OFFENSIVE CONTENT specifically
# (the task tier 6 barely touched): AUC 0.597->0.741.
TIER_8_FEATURES = TIER_6_FEATURES + ["german_toxicity_eistakovskii", "german_toxicity_ankekat"]

# Kept for backward compatibility with anything importing this name.
LIVE_FEATURES = FLOOR_FEATURES

HIGH_FPR_BUDGET = 0.05
MEDIUM_FPR_BUDGET = 0.20

# Per-class budgets are TIGHTER than the global ones on purpose. Three
# policies each firing at 5% would produce a union false-positive rate near
# 15%; 2.24% is the shared per-class budget that
# scripts/analyze_per_class_thresholds.py binary-searched to make the union
# land on the global policy's 5%, which is the only basis on which the two
# arms can be honestly compared. The MEDIUM budget is scaled by the same
# ratio (2.24/5) to keep the two tiers proportionate.
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
                    "docs/ENGINEERING_ASSESSMENT.md.",
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
    all_features = sorted({f for f in TIER_8_FEATURES if f != "anchors"})
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

    artifact = {
        # v3 adds `upgrade_tiers`. Every top-level field is exactly the
        # floor tier's own — an artifact with no upgrade_tiers key (or a
        # core/fusion.py that predates tiers) behaves identically to
        # every artifact before this version.
        "version": 3,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **floor,
        "upgrade_tiers": [tier_8, tier_6, tier_5],  # richest first
    }

    os.makedirs(os.path.dirname(ARTIFACT_FILE), exist_ok=True)
    with open(ARTIFACT_FILE, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact -> {ARTIFACT_FILE}")
    print(f"Floor: {FLOOR_FEATURES}")
    print("Upgrade tiers (richest first): eight_feature, six_feature, five_feature")


if __name__ == "__main__":
    main()
