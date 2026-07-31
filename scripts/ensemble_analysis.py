"""
Tests the project's central claim:

    An ensemble of specialised detectors under a learned, auditable fusion
    policy outperforms any single detector.

This is the thesis. It is also the product argument — if it holds, the gateway's
value is the fusion and governance layer rather than any one classifier, which
is a defensible position against vendors who ship a single model.

Runs on cached scores from `scripts.compare_detectors`, so it is fast and can be
iterated on freely.

METHODOLOGICAL COMMITMENTS
--------------------------
- OUT-OF-FOLD EVALUATION. The learned fusion is scored via stratified k-fold
  cross-validation using out-of-fold predictions only. Reporting in-sample
  performance for a trained model would guarantee it "wins" and would mean
  nothing.
- CONTAMINATED DETECTORS EXCLUDED BY DEFAULT. Two public detectors were trained
  on sources in our suite. Including them would let the ensemble inherit their
  memorisation. The default ensemble uses clean detectors only.
- SAME OPERATING POINT. Every configuration is compared at the threshold
  achieving the same FPR budget on the same rows.
- INTERPRETABILITY. Logistic-regression coefficients are reported. A fusion
  policy that cannot be explained is not auditable, and auditability is the
  product claim.

Usage:
    python -m scripts.ensemble_analysis
    python -m scripts.ensemble_analysis --include-contaminated
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.detectors import get_registry
from evaluation.metrics import (
    bootstrap_ci,
    confusion,
    derived,
    fmt_ci,
    recall_at_fpr,
    roc_auc,
    threshold_at_fpr,
)

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
REPORT_FILE = os.path.join("_evidence", "ensemble_analysis.json")
SEED = 20260723


def load_rows():
    rows = []
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_scores(name):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def evaluate(scores, labels, fpr_budget, n_boot, name):
    """Uniform metric bundle so every configuration is directly comparable."""
    auc = bootstrap_ci(scores, labels, roc_auc, n_boot=n_boot)
    rec = bootstrap_ci(scores, labels,
                       lambda s, lab: recall_at_fpr(s, lab, budget=fpr_budget),
                       n_boot=n_boot)
    thr = threshold_at_fpr(scores, labels, fpr_budget)
    cm = confusion(scores, labels, thr)
    return {"name": name, "auc": auc, "recall_at_fpr": rec,
            "threshold": thr, "confusion_matrix": cm, **derived(cm)}


def per_class_rates(rows, scores, threshold):
    out = {}
    for cls in ("prompt_injection", "jailbreak", "harmful_content"):
        sub = [s for r, s in zip(rows, scores) if r["attack_class"] == cls]
        if sub:
            out[cls] = {"n": len(sub),
                        "detected": sum(1 for s in sub if s >= threshold) / len(sub)}
    return out


def main():
    parser = argparse.ArgumentParser(description="Ensemble vs single detectors")
    parser.add_argument("--fpr-budget", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-contaminated", action="store_true",
                        help="Include detectors trained on suite sources (NOT a fair comparison).")
    args = parser.parse_args()

    rows = load_rows()
    registry = get_registry()

    # --- Assemble the feature matrix from cached detector scores ---
    usable, skipped = [], []
    for name, detector in registry.items():
        cached = load_scores(name)
        if cached is None:
            skipped.append((name, "no cached scores; run scripts.compare_detectors"))
            continue
        if detector.trained_on and not args.include_contaminated:
            skipped.append((name, f"trained on {list(detector.trained_on)} (excluded)"))
            continue
        if not all(r["id"] in cached for r in rows):
            skipped.append((name, "incomplete score cache"))
            continue
        usable.append((name, cached))

    print("Detectors in ensemble:")
    for name, _ in usable:
        print(f"  + {name:<24} targets={','.join(registry[name].targets)}")
    for name, reason in skipped:
        print(f"  - {name:<24} {reason}")

    if len(usable) < 2:
        raise SystemExit("\nNeed at least 2 detectors with cached scores. "
                         "Run: python -m scripts.compare_detectors")

    names = [n for n, _ in usable]
    X = np.array([[cached[r["id"]] for _, cached in usable] for r in rows])
    y = np.array([r["label"] for r in rows])

    print(f"\nRows: {len(rows)}  attacks: {int(y.sum())}  benign: {int((1 - y).sum())}")
    print(f"Features: {names}")

    results = []

    # --- Baseline: each detector alone ---
    for i, name in enumerate(names):
        results.append(evaluate(list(X[:, i]), list(y), args.fpr_budget,
                                args.bootstrap, f"single: {name}"))

    # --- Untrained ensemble: elementwise max (a plain OR of specialists) ---
    max_scores = list(X.max(axis=1))
    results.append(evaluate(max_scores, list(y), args.fpr_budget,
                            args.bootstrap, "ensemble: max (untrained)"))

    # --- Learned fusion, evaluated out-of-fold ---
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    results.append(evaluate(list(oof), list(y), args.fpr_budget,
                            args.bootstrap, "ensemble: learned (out-of-fold)"))

    # Coefficients from a full-data fit, for interpretation only — never for scoring.
    model.fit(X, y)
    logreg = model.named_steps["logisticregression"]
    coefs = dict(sorted(zip(names, logreg.coef_[0].tolist()),
                        key=lambda kv: -abs(kv[1])))

    # --- Report ---
    print(f"\n{'=' * 92}")
    print(f"ENSEMBLE vs SINGLE DETECTORS  (FPR budget {args.fpr_budget:.0%})")
    print("=" * 92)
    print(f"{'configuration':<36} {'AUC':>22} {'Recall@budget':>22}")
    for r in sorted(results, key=lambda x: -x["auc"]["point"]):
        print(f"{r['name']:<36} {fmt_ci(r['auc'], pct=False):>22} "
              f"{fmt_ci(r['recall_at_fpr']):>22}")

    best_single = max((r for r in results if r["name"].startswith("single")),
                      key=lambda r: r["auc"]["point"])
    learned = next(r for r in results if "learned" in r["name"])
    max_ens = next(r for r in results if "max" in r["name"])

    print(f"\n{'-' * 92}")
    print("THESIS TEST: does learned fusion beat the best single detector?")
    print(f"  best single      : {best_single['name']} "
          f"AUC {fmt_ci(best_single['auc'], pct=False)}")
    print(f"  max ensemble     : AUC {fmt_ci(max_ens['auc'], pct=False)}")
    print(f"  learned fusion   : AUC {fmt_ci(learned['auc'], pct=False)}")

    delta = learned["auc"]["point"] - best_single["auc"]["point"]
    overlap = not (learned["auc"]["lo"] > best_single["auc"]["hi"]
                   or best_single["auc"]["lo"] > learned["auc"]["hi"])
    print(f"\n  delta AUC        : {delta:+.4f}")
    if overlap:
        verdict = ("INCONCLUSIVE - confidence intervals overlap. The ensemble is "
                   "not demonstrably better than the best single detector on this data.")
    elif delta > 0:
        verdict = ("SUPPORTED - learned fusion beats the best single detector with "
                   "non-overlapping confidence intervals.")
    else:
        verdict = ("REFUTED - the best single detector beats learned fusion with "
                   "non-overlapping confidence intervals.")
    print(f"  verdict          : {verdict}")

    print(f"\n{'-' * 92}")
    print("FUSION COEFFICIENTS (standardised; sign = direction, magnitude = influence)")
    for name, c in coefs.items():
        bar = "#" * min(int(abs(c) * 20), 40)
        print(f"  {name:<26} {c:+.4f}  {bar}")

    print(f"\n{'-' * 92}")
    print("PER-CLASS DETECTION at each configuration's budget threshold")
    print(f"{'configuration':<36} {'injection':>12} {'jailbreak':>12} {'harmful':>12}")
    for r, sc in ((best_single, list(X[:, names.index(best_single['name'].split(': ')[1])])),
                  (max_ens, max_scores),
                  (learned, list(oof))):
        pc = per_class_rates(rows, sc, r["threshold"])
        cells = "".join(f"{pc.get(c, {}).get('detected', float('nan')):>12.1%}"
                        for c in ("prompt_injection", "jailbreak", "harmful_content"))
        print(f"{r['name']:<36}{cells}")

    payload = {
        "config": {
            "fpr_budget": args.fpr_budget,
            "bootstrap_resamples": args.bootstrap,
            "cv_folds": args.folds,
            "seed": SEED,
            "detectors": names,
            "excluded": [{"detector": n, "reason": r} for n, r in skipped],
            "contaminated_included": args.include_contaminated,
        },
        "results": results,
        "thesis": {
            "best_single": best_single["name"],
            "delta_auc_learned_vs_best_single": delta,
            "confidence_intervals_overlap": overlap,
            "verdict": verdict,
        },
        "fusion_coefficients": coefs,
        "fusion_intercept": float(logreg.intercept_[0]),
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
