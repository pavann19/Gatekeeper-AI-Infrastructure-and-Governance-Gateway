"""
Measures whether PER-CLASS fusion policies beat the single global policy,
BEFORE any of it is wired into the request path.

THE DESIGN PROBLEM THIS RESOLVES FIRST
--------------------------------------
"Give harmful_content its own threshold" is not directly implementable,
because at inference time the attack class is unknown — determining it is the
job being done. A per-class threshold can only be applied to a per-class
SCORE. So the real proposal is: train one policy per attack class (that class
as positive, benign as negative), score every prompt under all of them, and
take the most severe verdict. Each class then gets its own decision boundary,
which is what "harmful_content should be more sensitive without loosening
injection/jailbreak" actually requires.

THE COST THAT MUST BE MEASURED, NOT ASSUMED
-------------------------------------------
Three independent detectors each firing at a 5% false-positive budget do NOT
give a 5% combined budget — the union can approach 15%. Comparing per-class
recall at 5%-each against global recall at 5% total would be measuring a
looser operating point and calling it an improvement. So per-class thresholds
here are calibrated by binary-searching a shared per-class budget until the
UNION false-positive rate equals the global policy's 5%. Both arms are then
compared at genuinely the same FPR, which is the only comparison that means
anything.

Out-of-fold throughout: a policy scored on data it was fitted to would win by
memorisation.

Usage:
    python -m scripts.analyze_per_class_thresholds
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
REPORT_FILE = os.path.join("_evidence", "per_class_threshold_analysis.json")

FEATURES = ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"]
CLASSES = ["prompt_injection", "jailbreak", "harmful_content"]
FPR_BUDGET = 0.05
SEED = 20260723


def load_rows():
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_scores(name):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}; run: python -m scripts.compare_detectors")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def threshold_at_fpr(scores, labels, budget):
    """Lowest threshold whose FPR is within budget (maximising recall)."""
    neg = sorted((s for s, l in zip(scores, labels) if l == 0), reverse=True)
    if not neg:
        return float("inf")
    k = int(len(neg) * budget)
    return neg[k] if k < len(neg) else neg[-1]


def main():
    rows = load_rows()
    caches = {f: load_scores(f) for f in FEATURES}
    usable = [r for r in rows if all(r["id"] in caches[f] for f in FEATURES)]
    X = np.array([[caches[f][r["id"]] for f in FEATURES] for r in usable])
    y = np.array([r["label"] for r in usable])
    cls = np.array([r["attack_class"] for r in usable])
    print(f"{len(usable)} rows | {int(y.sum())} attacks | "
          f"{ {c: int((cls == c).sum()) for c in CLASSES} }")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def new_model():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    # ---- ARM 1: the current global policy, out-of-fold ----
    global_oof = cross_val_predict(new_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    global_thr = threshold_at_fpr(list(global_oof), list(y), FPR_BUDGET)
    global_flag = global_oof >= global_thr

    # ---- ARM 2: one policy per class ----
    # Each is trained on benign + that class only. Rows of OTHER attack classes
    # were never in its training set, so a full-fit model scores them without
    # leakage; in-training rows get out-of-fold scores.
    per_class_scores = {}
    for c in CLASSES:
        in_train = (y == 0) | (cls == c)
        Xc, yc = X[in_train], (cls[in_train] == c).astype(int)

        oof = cross_val_predict(new_model(), Xc, yc, cv=cv, method="predict_proba")[:, 1]
        full = new_model().fit(Xc, yc)
        scores = full.predict_proba(X)[:, 1]      # unseen rows: no leakage
        scores[in_train] = oof                     # seen rows: out-of-fold
        per_class_scores[c] = scores

    # Calibrate a SHARED per-class budget so the UNION FPR matches the global
    # arm's 5% — otherwise per-class is simply operating looser and any recall
    # gain is an artefact of that, not of the architecture.
    benign = y == 0

    def union_flags(alpha):
        flags = np.zeros(len(y), dtype=bool)
        thresholds = {}
        for c in CLASSES:
            s = per_class_scores[c]
            t = threshold_at_fpr(list(s), list(y), alpha)
            thresholds[c] = t
            flags |= s >= t
        return flags, thresholds

    lo, hi = 0.0, FPR_BUDGET
    for _ in range(40):
        mid = (lo + hi) / 2
        flags, _ = union_flags(mid)
        if flags[benign].mean() > FPR_BUDGET:
            hi = mid
        else:
            lo = mid
    per_alpha = lo
    per_flag, per_thresholds = union_flags(per_alpha)

    # ---- Compare at genuinely equal FPR ----
    def recalls(flag):
        out = {"overall": float(flag[y == 1].mean())}
        for c in CLASSES:
            out[c] = float(flag[cls == c].mean())
        return out

    g_rec, p_rec = recalls(global_flag), recalls(per_flag)
    g_fpr = float(global_flag[benign].mean())
    p_fpr = float(per_flag[benign].mean())

    print(f"\nglobal   FPR {g_fpr:.3%}  (threshold {global_thr:.4f})")
    print(f"per-class FPR {p_fpr:.3%}  (per-class budget {per_alpha:.4%})")
    print(f"  thresholds: { {c: round(t, 4) for c, t in per_thresholds.items()} }")

    print(f"\n{'metric':<20} {'global':>10} {'per-class':>11} {'delta':>9}")
    for k in ["overall"] + CLASSES:
        d = p_rec[k] - g_rec[k]
        print(f"{k:<20} {g_rec[k]:>9.1%} {p_rec[k]:>10.1%} {d:>+9.1%}")

    verdict = (
        "per-class WINS on harmful_content without losing overall"
        if p_rec["harmful_content"] > g_rec["harmful_content"] + 0.02
        and p_rec["overall"] >= g_rec["overall"] - 0.01
        else "per-class does NOT clearly beat the global policy at equal FPR"
    )
    print(f"\nVERDICT: {verdict}")

    os.makedirs("_evidence", exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "fpr_budget": FPR_BUDGET,
            "global": {"threshold": float(global_thr), "fpr": g_fpr, "recall": g_rec},
            "per_class": {"shared_alpha": per_alpha, "fpr": p_fpr,
                          "thresholds": {c: float(t) for c, t in per_thresholds.items()},
                          "recall": p_rec},
            "verdict": verdict,
        }, f, indent=2)
    print(f"Report -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
