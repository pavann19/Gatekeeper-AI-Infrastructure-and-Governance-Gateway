"""
Splits German performance by TASK, because the single "German AUC" figure
this project has been tracking turns out to average two unrelated problems
and therefore describes neither.

WHY THIS SPLIT EXISTS
---------------------
Expanding the suite for German volume pulled in philschmid/germeval18,
which is German social-media OFFENSIVE LANGUAGE (harmful_content). It is
2,931 of the 4,221 German rows -- 69% -- so a pooled "German AUC" is
dominated by it. The remaining German rows are PROMPT INJECTION, which is
what the roadmap's German-gap item was actually about (the original
benchmark was deepset/prompt-injections, German-heavy instruction
override).

Leave-one-source-out on the multilingual head made the conflation visible:
held out entirely, it scores 0.982 / 0.946 / 0.941 AUC on the three unseen
German INJECTION sources and 0.615 on germeval18. Averaging those into one
number produces a figure that overstates the injection problem and
understates the offensive-content one.

This script reports the deployed and candidate fusions on each German task
separately, so the two can be tracked -- and fixed -- independently.

Usage:
    python -m scripts.analyze_german_by_task
"""
import json
import os

import numpy as np

from evaluation.metrics import bootstrap_ci, fmt_ci, recall_at_fpr, roc_auc
from scripts.sweep_fusion_variants import (
    ALL_7,
    DEPLOYED_4,
    PLUS_BOTH_6,
    load_scores,
    oof,
)

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
REPORT_FILE = os.path.join("_evidence", "german_by_task.json")
FPR_BUDGET = 0.05
N_BOOT = 500

# Sources whose German rows are prompt-injection-vs-benign. Everything else
# German is offensive-content-vs-benign (germeval18, plus a negligible
# toxic-chat tail).
INJECTION_SOURCES = {
    "deepset/prompt-injections",
    "rikka-snow/prompt-injection-multilingual",
    "Octavio-Santana/prompt-injection-attack-detection-multilingual",
    "lakera/gandalf_ignore_instructions",
}


def main():
    rows_all = [json.loads(line) for line in open(SUITE_FILE, "r", encoding="utf-8")]
    caches = {n: load_scores(n) for n in ALL_7}
    rows = [r for r in rows_all if all(r["id"] in caches[n] for n in ALL_7)]
    y = np.array([r["label"] for r in rows])

    de = [i for i, r in enumerate(rows) if r["language"] == "de"]
    de_inj = [i for i in de if rows[i]["source"] in INJECTION_SOURCES]
    de_off = [i for i in de if rows[i]["source"] not in INJECTION_SOURCES]
    print(f"German injection-task rows: {len(de_inj)} "
          f"({int(sum(y[i] for i in de_inj))} attacks)")
    print(f"German offensive-task rows: {len(de_off)} "
          f"({int(sum(y[i] for i in de_off))} attacks)")

    report = {"n_de_injection": len(de_inj), "n_de_offensive": len(de_off),
              "fpr_budget": FPR_BUDGET, "variants": {}}

    for tag, feats in (("deployed_4", DEPLOYED_4),
                       ("plus_both_6", PLUS_BOTH_6),
                       ("all_7", ALL_7)):
        scores = oof(feats, rows, caches, y)
        print(f"\n[{tag}]")
        entry = {}
        for name, idx in (("german_injection", de_inj), ("german_offensive", de_off)):
            s = [scores[i] for i in idx]
            lab = [int(y[i]) for i in idx]
            auc = bootstrap_ci(s, lab, roc_auc, n_boot=N_BOOT)
            rec = bootstrap_ci(s, lab,
                               lambda a, b: recall_at_fpr(a, b, budget=FPR_BUDGET),
                               n_boot=N_BOOT)
            print(f"   {name:<20} n={len(idx):<6} AUC={fmt_ci(auc, pct=False):<24} "
                  f"recall@{FPR_BUDGET:.0%}FPR={fmt_ci(rec)}")
            entry[name] = {"n": len(idx), "n_attack": int(sum(lab)),
                           "auc": auc, "recall_at_fpr": rec}
        report["variants"][tag] = entry

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
