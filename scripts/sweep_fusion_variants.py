"""
Sweeps fusion variants against the full suite, out-of-fold, reporting
per-language -- the experiment that actually tries to CLOSE the German gap
rather than measure it again.

Variants, each a fresh 5-fold OOF logistic regression (same seed/folds as
scripts.ensemble_analysis, so numbers are comparable to every prior
ensemble result in this project):

  deployed_4      the live feature set, baseline to beat
  plus_deepset_5  + deepset_injection as a FIFTH feature (NOT a swap --
                  the swap was tested and decisively rejected; the fusion
                  had learned correlation structure specific to
                  protectai_injection, so substituting it unretrained
                  destroyed pooled and English performance)
  de_conditional  the deployed 4 features, but the logistic regression is
                  FIT ONLY ON GERMAN ROWS and applied only to German rows.
                  Directly targets the finding that one global weighting
                  cannot serve two populations whose feature reliability
                  differs -- toxic_bert is near-useless on German while
                  carrying real signal on English, and a single set of
                  coefficients has to compromise between those.
  de_conditional_5  the same, with deepset_injection added.

WHY LANGUAGE-CONDITIONAL IS THE INTERESTING ONE
------------------------------------------------
Every earlier attempt treated the German gap as a THRESHOLD problem (move
the cut) or a FEATURE problem (swap the detector). Both failed. The
measured evidence points at neither: German OOF AUC 0.671 vs English 0.927
on identical features means the fusion's WEIGHTS -- not its inputs, not its
cutoff -- are fit to a population German traffic isn't drawn from. A
per-language weighting is the first variant here that addresses that
directly, and it costs nothing extra at inference time: the language tag is
already computed, and it selects between two tiny coefficient vectors.

Usage:
    python -m scripts.sweep_fusion_variants
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
REPORT_FILE = os.path.join("_evidence", "fusion_variant_sweep.json")
SEED = 20260723
FOLDS = 5
FPR_BUDGET = 0.05
N_BOOT = 500

DEPLOYED_4 = ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"]
PLUS_DEEPSET_5 = DEPLOYED_4 + ["deepset_injection"]
PLUS_PG2_5 = DEPLOYED_4 + ["prompt_guard_2"]
PLUS_BOTH_6 = DEPLOYED_4 + ["deepset_injection", "prompt_guard_2"]

# Every feature needed by any variant, so one coverage filter serves all of
# them and every variant is scored on exactly the same rows (otherwise a
# variant using a thinner-cached feature would be measured on a different,
# quietly easier population).
ALL_FEATURES = DEPLOYED_4 + ["deepset_injection", "prompt_guard_2"]


def load_scores(name):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def oof(features, rows, caches, y):
    X = np.array([[caches[n][r["id"]] for n in features] for r in rows])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def summarize(scores, labels, tag):
    auc = bootstrap_ci(list(scores), list(labels), roc_auc, n_boot=N_BOOT)
    rec = bootstrap_ci(list(scores), list(labels),
                       lambda s, lab: recall_at_fpr(s, lab, budget=FPR_BUDGET), n_boot=N_BOOT)
    print(f"    {tag:<10} n={len(labels):<6} AUC={fmt_ci(auc, pct=False):<24} "
          f"recall@{FPR_BUDGET:.0%}FPR={fmt_ci(rec)}")
    return {"n": len(labels), "n_attack": int(sum(labels)), "auc": auc, "recall_at_fpr": rec}


def main():
    rows_all = [json.loads(line) for line in open(SUITE_FILE, "r", encoding="utf-8")]
    caches = {n: load_scores(n) for n in ALL_FEATURES}
    rows = [r for r in rows_all if all(r["id"] in caches[n] for n in ALL_FEATURES)]
    print(f"Usable rows: {len(rows)} of {len(rows_all)}")
    y = np.array([r["label"] for r in rows])
    langs = [r["language"] for r in rows]
    report = {"seed": SEED, "folds": FOLDS, "fpr_budget": FPR_BUDGET, "n_rows": len(rows),
              "variants": {}}

    # --- Pooled variants: one model over all languages -------------------
    for tag, feats in (("deployed_4", DEPLOYED_4),
                       ("plus_deepset_5", PLUS_DEEPSET_5),
                       ("plus_pg2_5", PLUS_PG2_5),
                       ("plus_both_6", PLUS_BOTH_6)):
        print(f"\n[{tag}] features={feats}")
        scores = oof(feats, rows, caches, y)
        entry = {"features": feats, "trained_on": "all languages", "by_language": {}}
        entry["pooled"] = summarize(scores, y, "pooled")
        for lang in ("en", "de", "other"):
            idx = [i for i, la in enumerate(langs) if la == lang]
            if len(idx) < 50:
                continue
            entry["by_language"][lang] = summarize(
                [scores[i] for i in idx], [int(y[i]) for i in idx], lang)
        report["variants"][tag] = entry

    # --- Language-conditional: fit ONLY on German, apply ONLY to German ---
    de_idx = [i for i, la in enumerate(langs) if la == "de"]
    de_rows = [rows[i] for i in de_idx]
    y_de = np.array([int(y[i]) for i in de_idx])
    for tag, feats in (("de_conditional", DEPLOYED_4),
                       ("de_conditional_5", PLUS_DEEPSET_5),
                       ("de_conditional_6", PLUS_BOTH_6)):
        print(f"\n[{tag}] features={feats}  (fit on German rows only)")
        scores_de = oof(feats, de_rows, caches, y_de)
        entry = {"features": feats, "trained_on": "German rows only", "by_language": {}}
        entry["by_language"]["de"] = summarize(scores_de, y_de, "de")
        report["variants"][tag] = entry

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
