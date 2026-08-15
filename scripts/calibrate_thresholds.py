"""
Threshold calibration for the Gatekeeper risk pipeline.

WHY THIS EXISTS
---------------
Every decision threshold in core/config.py (SEMANTIC_THRESHOLD_HIGH = 0.48,
META_INTENT_THRESHOLD = 0.40, ...) was originally chosen by intuition. That is
indefensible in a technical report and unsafe in a product: a threshold is an
operating point on a precision/recall trade-off curve, and it must be selected
against a stated objective, not guessed.

This script replaces "I picked 0.48" with:

    "0.48 is the operating point that maximises recall subject to a
     false-positive rate of at most 5% on a held-out split."

METHOD
------
1. Extract raw signals ONCE per prompt (the expensive part: embedding +
   FAISS search + meta-intent similarity). Results are cached to disk so
   subsequent sweeps are instantaneous.
2. Split into calibration / holdout sets with a fixed seed.
3. Sweep the threshold grid over the calibration set, evaluating the same
   deterministic fusion rule the live pipeline uses.
4. Select the operating point satisfying the FPR budget, then report its
   performance on the HOLDOUT set — the number that goes in the report.
5. Emit ROC/PR curves and AUC for the continuous signal.

The semantic judge (Stage 4) is deliberately excluded from the sweep: it is
non-deterministic and expensive, so sweeping over it is not reproducible. The
calibration therefore measures the *deterministic* pipeline, and the report
should state that explicitly. Judge contribution belongs in an ablation study.

Usage:
    python -m scripts.calibrate_thresholds
    python -m scripts.calibrate_thresholds --fpr-budget 0.02
    python -m scripts.calibrate_thresholds --refresh-signals
"""
import argparse
import json
import os
import random

from core.embeddings import get_embedding
from core.risk import (
    _ensure_faiss_initialized,
    check_meta_intent,
    hard_ban_triggered,
)
from core.vector_store import threat_store

EVIDENCE_DIR = "_evidence"
SIGNALS_FILE = os.path.join(EVIDENCE_DIR, "calibration_signals.json")
REPORT_FILE = os.path.join(EVIDENCE_DIR, "calibration_report.json")
CURVE_FILE = os.path.join(EVIDENCE_DIR, "roc_curve.png")

HOLDOUT_FRACTION = 0.3
SEED = 20260723


# ---------------------------------------------------------------------------
# Stage 1 — signal extraction (expensive, cached)
# ---------------------------------------------------------------------------

def extract_signals(limit=None, refresh=False):
    """
    Runs each prompt through the deterministic signal collectors and caches
    the raw scores. This is the only step that touches the ML models.
    """
    if os.path.exists(SIGNALS_FILE) and not refresh:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} cached signal records from {SIGNALS_FILE}")
        print("(pass --refresh-signals to recompute)")
        return records

    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    prompts, labels = ds["text"], ds["label"]
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    _ensure_faiss_initialized()

    records = []
    print(f"Extracting signals for {len(prompts)} prompts (one-time cost)...")
    for prompt, label in tqdm(zip(prompts, labels), total=len(prompts)):
        symbolic, detail = hard_ban_triggered(prompt)
        vec = get_embedding(prompt)
        records.append({
            "prompt": prompt,
            "is_malicious": bool(label == 1),
            "symbolic_triggered": bool(symbolic),
            "symbolic_detail": detail,
            "threat_score": float(threat_store.get_max_similarity(vec)),
            "meta_intent_score": float(check_meta_intent(vec)),
        })

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Cached signals -> {SIGNALS_FILE}")
    return records


# ---------------------------------------------------------------------------
# Stage 2 — the fusion rule, mirrored from core/risk.py:fuse_signals
# ---------------------------------------------------------------------------

def predict(rec, t_high, t_med, t_meta):
    """
    Deterministic prediction under a candidate threshold set.

    Mirrors fuse_signals() minus the judge and the domain guardrail (the
    latter is a scoping decision and is not part of safety calibration).
    Returns True if the prompt would be flagged (HIGH or MEDIUM).
    """
    if rec["symbolic_triggered"]:
        return True
    if rec["meta_intent_score"] >= t_meta:
        return True
    return rec["threat_score"] >= t_med


def confusion(records, t_high, t_med, t_meta):
    tp = fp = tn = fn = 0
    for r in records:
        pred = predict(r, t_high, t_med, t_meta)
        if r["is_malicious"] and pred:
            tp += 1
        elif not r["is_malicious"] and pred:
            fp += 1
        elif not r["is_malicious"] and not pred:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def metrics(records, t_high, t_med, t_meta):
    tp, fp, tn, fn = confusion(records, t_high, t_med, t_meta)
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold_high": round(t_high, 4),
        "threshold_medium": round(t_med, 4),
        "threshold_meta_intent": round(t_meta, 4),
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


# ---------------------------------------------------------------------------
# Stage 3 — ROC / PR curves on the continuous signal
# ---------------------------------------------------------------------------

def continuous_score(rec):
    """
    Collapses the deterministic signals into one score for ROC analysis.
    Symbolic hits are a deterministic veto and score 1.0 by construction.
    """
    if rec["symbolic_triggered"]:
        return 1.0
    return max(rec["threat_score"], rec["meta_intent_score"])


def roc_curve(records):
    """Returns (points, auc). Each point is (fpr, tpr, threshold)."""
    scored = sorted(((continuous_score(r), r["is_malicious"]) for r in records),
                    key=lambda x: -x[0])
    n_pos = sum(1 for _, m in scored if m)
    n_neg = len(scored) - n_pos
    if not n_pos or not n_neg:
        return [], 0.0

    points = [(0.0, 0.0, 1.01)]
    tp = fp = 0
    for score, is_mal in scored:
        if is_mal:
            tp += 1
        else:
            fp += 1
        points.append((fp / n_neg, tp / n_pos, score))
    points.append((1.0, 1.0, -0.01))

    # Trapezoidal AUC
    auc = 0.0
    for (x0, y0, _), (x1, y1, _) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2
    return points, auc


def pr_curve(records):
    scored = sorted(((continuous_score(r), r["is_malicious"]) for r in records),
                    key=lambda x: -x[0])
    n_pos = sum(1 for _, m in scored if m)
    if not n_pos:
        return []
    points = []
    tp = fp = 0
    for score, is_mal in scored:
        if is_mal:
            tp += 1
        else:
            fp += 1
        points.append((tp / n_pos, tp / (tp + fp), score))  # (recall, precision, threshold)
    return points


def plot_curves(roc_points, auc, pr_points):
    """Renders ROC + PR curves if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([p[0] for p in roc_points], [p[1] for p in roc_points],
             linewidth=2, label=f"Gatekeeper (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC — deterministic signals")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    if pr_points:
        ax2.plot([p[0] for p in pr_points], [p[1] for p in pr_points], linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    fig.savefig(CURVE_FILE, dpi=150)
    print(f"Curves -> {CURVE_FILE}")
    return CURVE_FILE


# ---------------------------------------------------------------------------
# Stage 4 — the sweep
# ---------------------------------------------------------------------------

def frange(start, stop, step):
    values, v = [], start
    while v <= stop + 1e-9:
        values.append(round(v, 4))
        v += step
    return values


def sweep(records, fpr_budget):
    """
    Grid-searches thresholds, returning all evaluated points plus the
    selection that maximises recall subject to fpr <= budget.
    """
    grid = []
    for t_med in frange(0.05, 0.70, 0.01):
        for t_meta in frange(0.20, 0.80, 0.05):
            t_high = max(t_med, 0.48)
            grid.append(metrics(records, t_high, t_med, t_meta))

    feasible = [m for m in grid if m["fpr"] <= fpr_budget]
    if feasible:
        # Maximise recall; break ties on precision, then on the looser threshold.
        best = max(feasible, key=lambda m: (m["recall"], m["precision"]))
        rationale = f"max recall subject to FPR <= {fpr_budget:.1%}"
    else:
        best = min(grid, key=lambda m: m["fpr"])
        rationale = (f"NO operating point satisfies FPR <= {fpr_budget:.1%}; "
                     f"reporting the minimum-FPR point instead")

    best_f1 = max(grid, key=lambda m: m["f1"])
    return grid, best, best_f1, rationale


def main():
    parser = argparse.ArgumentParser(description="Calibrate Gatekeeper thresholds")
    parser.add_argument("--fpr-budget", type=float, default=0.05,
                        help="Maximum acceptable false-positive rate (default 0.05).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N prompts.")
    parser.add_argument("--refresh-signals", action="store_true",
                        help="Recompute cached signals (slow).")
    args = parser.parse_args()

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    records = extract_signals(limit=args.limit, refresh=args.refresh_signals)

    # Deterministic split.
    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - HOLDOUT_FRACTION))
    calib, holdout = shuffled[:split], shuffled[split:]

    n_mal = sum(1 for r in records if r["is_malicious"])
    print(f"\nDataset: {len(records)} prompts ({n_mal} malicious, {len(records) - n_mal} benign)")
    print(f"Split:   {len(calib)} calibration / {len(holdout)} holdout (seed={SEED})")

    grid, best, best_f1, rationale = sweep(calib, args.fpr_budget)

    # The number that goes in the report: chosen on calibration, measured on holdout.
    holdout_perf = metrics(
        holdout,
        best["threshold_high"], best["threshold_medium"], best["threshold_meta_intent"],
    )

    roc_points, auc = roc_curve(records)
    pr_points = pr_curve(records)
    plot_path = plot_curves(roc_points, auc, pr_points)

    print(f"\n=== Selected operating point ({rationale}) ===")
    print(f"  SEMANTIC_THRESHOLD_MEDIUM = {best['threshold_medium']}")
    print(f"  SEMANTIC_THRESHOLD_HIGH   = {best['threshold_high']}")
    print(f"  META_INTENT_THRESHOLD     = {best['threshold_meta_intent']}")
    print(f"\n  Calibration set: recall={best['recall']:.2%}  precision={best['precision']:.2%}  "
          f"FPR={best['fpr']:.2%}  F1={best['f1']:.3f}")
    print(f"  HOLDOUT set:     recall={holdout_perf['recall']:.2%}  "
          f"precision={holdout_perf['precision']:.2%}  "
          f"FPR={holdout_perf['fpr']:.2%}  F1={holdout_perf['f1']:.3f}")
    print(f"  Holdout confusion: {holdout_perf['confusion_matrix']}")
    print(f"\n  ROC AUC (full dataset, deterministic signals): {auc:.3f}")
    print(f"\n  Best-F1 point (ignoring FPR budget): "
          f"t_med={best_f1['threshold_medium']} F1={best_f1['f1']:.3f} FPR={best_f1['fpr']:.2%}")

    report = {
        "method": {
            "objective": rationale,
            "fpr_budget": args.fpr_budget,
            "split": {"calibration": len(calib), "holdout": len(holdout), "seed": SEED},
            "judge_excluded": True,
            "judge_exclusion_rationale":
                "Stage 4 arbitration is non-deterministic and expensive; sweeping over "
                "it is not reproducible. Calibration covers the deterministic pipeline "
                "only. Judge contribution belongs in a separate ablation study.",
            "domain_guardrail_excluded": True,
            "domain_guardrail_exclusion_rationale":
                "Topicality is a scoping decision, not a safety decision, and is not "
                "part of the safety confusion matrix.",
        },
        "dataset": {
            "name": "deepset/prompt-injections",
            "n": len(records),
            "malicious": n_mal,
            "benign": len(records) - n_mal,
        },
        "selected_operating_point": best,
        "holdout_performance": holdout_perf,
        "best_f1_point": best_f1,
        "roc_auc": auc,
        "roc_curve": [{"fpr": round(f, 5), "tpr": round(t, 5), "threshold": round(s, 5)}
                      for f, t, s in roc_points],
        "sweep": grid,
        "plot": plot_path,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report -> {REPORT_FILE}")
    print("\nApply the selected point by setting these in .env or core/config.py.")


if __name__ == "__main__":
    main()
