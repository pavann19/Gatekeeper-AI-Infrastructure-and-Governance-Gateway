"""
Runs the DEPLOYED 4-feature fusion's out-of-fold, by-language breakdown
against the FULL eval suite and writes it to disk -- a thin wrapper around
scripts/analyze_multilingual_fusion.py's own helper functions, kept
separate rather than editing that script, because that script's own
CANDIDATE (+prompt_guard_2) comparison requires prompt_guard_2 scores that
are not (and don't need to be, for this question) kept at full coverage --
prompt_guard_2 is gated, not deployed, and already established in §1x as
not moving pooled performance. This script answers only "how does what is
actually deployed behave, per language, on the full suite" without that
dependency.

Usage:
    python -m scripts.analyze_fusion_by_language_full
"""
import json
import os

import numpy as np

from evaluation.metrics import threshold_at_fpr
from scripts.analyze_multilingual_fusion import DEPLOYED, by_language, load_scores, oof_scores

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
REPORT_FILE = os.path.join("_evidence", "fusion_by_language_full_suite.json")
FPR_BUDGET = 0.05


def main():
    rows = [json.loads(line) for line in open(SUITE_FILE, "r", encoding="utf-8")]
    caches = {n: load_scores(n) for n in DEPLOYED}
    usable = [r for r in rows if all(r["id"] in caches[n] for n in DEPLOYED)]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"Dropping {dropped} rows missing a cached score for one or more features.")

    y = np.array([r["label"] for r in usable])
    oof = oof_scores(DEPLOYED, usable, caches, y)
    by_lang = by_language(oof, usable, y)

    pooled_threshold = threshold_at_fpr(list(oof), list(y), FPR_BUDGET)
    langs = [r["language"] for r in usable]
    labels_list = list(y)

    per_language_at_pooled_threshold = {}
    for lang in ("en", "de", "other"):
        idx = [i for i, la in enumerate(langs) if la == lang]
        if not idx:
            continue
        labs = [labels_list[i] for i in idx]
        scores = [oof[i] for i in idx]
        n_benign = max(1, sum(1 for lab in labs if lab == 0))
        n_attack = max(1, sum(labs))
        fpr = sum(1 for s, lab in zip(scores, labs) if lab == 0 and s >= pooled_threshold) / n_benign
        recall = sum(1 for s, lab in zip(scores, labs) if lab == 1 and s >= pooled_threshold) / n_attack
        per_language_at_pooled_threshold[lang] = {"n": len(idx), "fpr": fpr, "recall": recall}
        print(f"{lang}: n={len(idx)} FPR@pooled_thr={fpr:.1%} recall@pooled_thr={recall:.1%}")

    report = {
        "n_rows": len(usable),
        "n_dropped": dropped,
        "fpr_budget": FPR_BUDGET,
        "deployed_features": DEPLOYED,
        "oof_auc_and_recall_by_language": by_lang,
        "pooled_threshold_at_fpr_budget": pooled_threshold,
        "per_language_at_pooled_threshold": per_language_at_pooled_threshold,
    }
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
