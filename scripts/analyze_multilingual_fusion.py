"""
Tests two multilingual hypotheses against the DEPLOYED 4-detector fusion
ensemble, out-of-fold, broken out by language — the Phase 1 "multilingual
encoder" item in docs/ROADMAP_V2.md.

WHY THIS SCRIPT EXISTS
-----------------------
docs/ENGINEERING_ASSESSMENT.md §1b/§1d already established, at the SINGLE
DETECTOR level, that swapping `all-mpnet-base-v2` for a multilingual encoder
is no longer on the critical path for the German gap — adopting
`protectai_injection` (German AUC 0.872 standalone) already closes most of
it, and `prompt_guard_2` (German AUC 0.970 standalone, once Meta's gated
licence is accepted) closes it further. Neither of those numbers was ever
measured for the ENSEMBLE actually deployed in `core/fusion.py` — this
script asks two concrete questions that decide what to build next:

  1. How well does the DEPLOYED 4-feature fusion (anchors, protectai_
     injection, madhurjindal_jailbreak, toxic_bert) actually perform on
     German prompts, out-of-fold, compared to English? (Nobody had measured
     the ENSEMBLE by language before — only single detectors.)
  2. Does adding prompt_guard_2 as a 5th feature move that number, enough to
     justify the operational cost of a per-deployment Meta licence
     acceptance that `fused_threat_score` has no graceful per-feature
     fallback for (ANY missing required feature drops the whole ensemble to
     the anchors-only fallback path, not just the German-specific gain)?

METHODOLOGY
-----------
Out-of-fold only (StratifiedKFold, same seed/fold count as
scripts.ensemble_analysis) — an in-sample number here would be exactly the
kind of misleading result §1b already warned against. Same cached detector
scores as scripts.train_fusion_policy and scripts.ensemble_analysis, so
this is fast and reproducible without re-running any transformer inference.

Usage:
    python -m scripts.analyze_multilingual_fusion
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import bootstrap_ci, fmt_ci, recall_at_fpr, roc_auc

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
REPORT_FILE = os.path.join("_evidence", "multilingual_fusion_analysis.json")
SEED = 20260723
FOLDS = 5
FPR_BUDGET = 0.05
N_BOOT = 500

DEPLOYED = ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"]
CANDIDATE = DEPLOYED + ["prompt_guard_2"]


def load_scores(name):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def oof_scores(feature_names, usable, caches, y):
    X = np.array([[caches[n][r["id"]] for n in feature_names] for r in usable])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def by_language(oof, usable, y):
    langs = [r["language"] for r in usable]
    out = {}
    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        if len(idx) < 10:  # too thin to bootstrap meaningfully
            continue
        scores = [oof[i] for i in idx]
        labels = [y[i] for i in idx]
        out[lang] = {
            "n": len(idx),
            "n_attack": int(sum(labels)),
            "auc": bootstrap_ci(scores, labels, roc_auc, n_boot=N_BOOT),
            "recall_at_fpr": bootstrap_ci(
                scores, labels, lambda s, l: recall_at_fpr(s, l, budget=FPR_BUDGET), n_boot=N_BOOT
            ),
        }
    return out


def overlap(ci_a, ci_b):
    return not (ci_a["lo"] > ci_b["hi"] or ci_b["lo"] > ci_a["hi"])


def main():
    rows = [json.loads(l) for l in open(SUITE_FILE, "r", encoding="utf-8")]
    caches = {n: load_scores(n) for n in CANDIDATE}
    usable = [r for r in rows if all(r["id"] in caches[n] for n in CANDIDATE)]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"Dropping {dropped} rows missing a cached score for one or more features.")
    y = np.array([r["label"] for r in usable])

    print(f"Rows: {len(usable)}  attacks: {int(y.sum())}  benign: {int((1 - y).sum())}")

    oof_deployed = oof_scores(DEPLOYED, usable, caches, y)
    oof_candidate = oof_scores(CANDIDATE, usable, caches, y)

    deployed_by_lang = by_language(oof_deployed, usable, y)
    candidate_by_lang = by_language(oof_candidate, usable, y)

    print(f"\n{'=' * 78}")
    print("DEPLOYED (4-feature) fusion, out-of-fold, by language")
    print("=" * 78)
    for lang, r in sorted(deployed_by_lang.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {lang:<8} n={r['n']:<6} AUC={fmt_ci(r['auc'], pct=False):<22} "
              f"recall@{FPR_BUDGET:.0%}FPR={fmt_ci(r['recall_at_fpr'])}")

    en, de = deployed_by_lang.get("en"), deployed_by_lang.get("de")
    if en and de:
        decisive = not overlap(en["auc"], de["auc"])
        print(f"\n  English vs German AUC overlap: {'NO — decisive gap' if decisive else 'yes — inconclusive'}")

    print(f"\n{'=' * 78}")
    print("CANDIDATE (+ prompt_guard_2) fusion, out-of-fold, by language")
    print("=" * 78)
    for lang, r in sorted(candidate_by_lang.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {lang:<8} n={r['n']:<6} AUC={fmt_ci(r['auc'], pct=False):<22} "
              f"recall@{FPR_BUDGET:.0%}FPR={fmt_ci(r['recall_at_fpr'])}")

    print(f"\n{'-' * 78}")
    print("DOES ADDING prompt_guard_2 MOVE GERMAN PERFORMANCE?")
    de_candidate = candidate_by_lang.get("de")
    if de and de_candidate:
        delta = de_candidate["auc"]["point"] - de["auc"]["point"]
        decisive = not overlap(de["auc"], de_candidate["auc"])
        print(f"  German AUC: {fmt_ci(de['auc'], pct=False)} -> {fmt_ci(de_candidate['auc'], pct=False)}"
              f"  (delta {delta:+.4f}, {'DECISIVE' if decisive else 'NOT decisive — CIs overlap'})")

    pooled_deployed = bootstrap_ci(list(oof_deployed), list(y), roc_auc, n_boot=N_BOOT)
    pooled_candidate = bootstrap_ci(list(oof_candidate), list(y), roc_auc, n_boot=N_BOOT)
    pooled_decisive = not overlap(pooled_deployed, pooled_candidate)
    print(f"\n  Pooled AUC: {fmt_ci(pooled_deployed, pct=False)} -> {fmt_ci(pooled_candidate, pct=False)}"
          f"  ({'DECISIVE' if pooled_decisive else 'NOT decisive — CIs overlap'})")

    report = {
        "seed": SEED, "folds": FOLDS, "fpr_budget": FPR_BUDGET, "n_bootstrap": N_BOOT,
        "n_rows": len(usable), "n_attack": int(y.sum()), "n_benign": int((1 - y).sum()),
        "deployed_features": DEPLOYED, "candidate_features": CANDIDATE,
        "deployed_by_language": deployed_by_lang,
        "candidate_by_language": candidate_by_lang,
        "pooled_deployed_auc": pooled_deployed,
        "pooled_candidate_auc": pooled_candidate,
        "pooled_decisive": pooled_decisive,
    }
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
