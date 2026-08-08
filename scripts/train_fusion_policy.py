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

WHY ONLY FOUR DETECTORS
-----------------------
The full comparison found Prompt Guard 2 and Llama Guard measurably strengthen
the ensemble, but both are Meta-gated: a fresh deployment needs its own licence
acceptance before either loads, and Llama Guard is far too slow for a live
request path (~seconds, not milliseconds, on CPU). The live fusion is
deliberately restricted to detectors that are (a) unencumbered by external
licensing and (b) fast enough for synchronous request handling:

    anchors, protectai_injection, madhurjindal_jailbreak, toxic_bert

This is exactly the four-detector pool from the FIRST ensemble_analysis run,
which showed AUC 0.944 [0.936, 0.951] against the best single detector's 0.909
[0.899, 0.919] — non-overlapping, validated. Prompt Guard 2 / Llama Guard remain
available as an opt-in upgrade for deployments that complete the licence step;
wiring that in is future work, not this script's job.

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

# Order matters: this list IS the feature order baked into the artifact.
# core/fusion.py must score detectors in this exact order at inference time.
LIVE_FEATURES = ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"]

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


def main():
    rows = load_suite()
    caches = {name: load_scores(name) for name in LIVE_FEATURES}

    # Every row must have a score from every live feature — no detector here
    # has a contamination exclusion (none of the four were trained on a suite
    # source), so nothing needs to be dropped.
    usable = [r for r in rows if all(r["id"] in caches[f] for f in LIVE_FEATURES)]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"Dropping {dropped} rows missing a cached score for one or more features.")

    X = [[caches[f][r["id"]] for f in LIVE_FEATURES] for r in usable]
    y = [r["label"] for r in usable]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_scaled, y)

    # Threshold selection operates on the PROBABILITY output, on the same data
    # the model was fit on. This is the deploy-time artifact, not an unbiased
    # accuracy estimate — the unbiased estimate is scripts.ensemble_analysis's
    # out-of-fold AUC (0.944), already reported in the engineering assessment.
    probs = model.predict_proba(X_scaled)[:, 1]
    auc_insample = roc_auc(list(probs), y)
    threshold_high = threshold_at_fpr(list(probs), y, budget=HIGH_FPR_BUDGET)
    threshold_medium = threshold_at_fpr(list(probs), y, budget=MEDIUM_FPR_BUDGET)

    # ---- PER-CLASS POLICIES ----
    # One policy per attack class (that class positive, benign negative), so
    # each gets its own decision boundary. This is what "harmful_content
    # should be more sensitive without loosening injection/jailbreak"
    # actually requires: a per-class threshold is only meaningful against a
    # per-class SCORE, since at inference the class is unknown — determining
    # it is the job being done.
    #
    # Budgets are per-class, so three policies each at 5% would give a UNION
    # false-positive rate approaching 15%, not 5%. PER_CLASS_*_FPR_BUDGET are
    # therefore set to the shared budget that
    # scripts/analyze_per_class_thresholds.py binary-searched to make the
    # union match the global policy's 5% — measured at equal FPR, per-class
    # improved harmful_content recall 28.7% -> 31.5% out-of-fold, with
    # overall recall not regressing. Anything else would be comparing a
    # looser operating point and calling the difference an improvement.
    classes = sorted({r["attack_class"] for r in usable if r["label"] == 1})
    per_class = {}
    for cls in classes:
        idx = [i for i, r in enumerate(usable)
               if r["label"] == 0 or r["attack_class"] == cls]
        Xc = [X[i] for i in idx]
        yc = [1 if usable[i]["attack_class"] == cls else 0 for i in idx]

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

    artifact = {
        # v2 adds `per_class`. The top-level global policy fields are kept
        # verbatim so an older core/fusion.py, or a deployment that disables
        # per-class scoring, still loads and behaves exactly as before.
        "version": 2,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": LIVE_FEATURES,
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
                    "estimate is scripts.ensemble_analysis's out-of-fold AUC: "
                    "0.944 [0.936, 0.951] against this exact 4-detector pool.",
        },
    }

    os.makedirs(os.path.dirname(ARTIFACT_FILE), exist_ok=True)
    with open(ARTIFACT_FILE, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Trained on {len(usable)} rows ({sum(y)} attack / {len(y) - sum(y)} benign)")
    print(f"Features (order matters): {LIVE_FEATURES}")
    print(f"Coefficients: {dict(zip(LIVE_FEATURES, artifact['coefficients']))}")
    print(f"Intercept: {artifact['intercept']:.4f}")
    print(f"threshold_high   (<= {HIGH_FPR_BUDGET:.0%} FPR): {threshold_high:.4f}")
    print(f"threshold_medium (<= {MEDIUM_FPR_BUDGET:.0%} FPR): {threshold_medium:.4f}")
    print(f"In-sample AUC (sanity check only): {auc_insample:.4f}")
    print(f"\nArtifact -> {ARTIFACT_FILE}")


if __name__ == "__main__":
    main()
