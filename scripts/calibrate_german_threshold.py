"""
Fits and evaluates a German-specific decision threshold for
`protectai_injection`, out-of-sample and against the contamination-excluded
population -- the "German-specific detection gap" item in
docs/ROADMAP_V2.md Phase 1, previously blocked on data volume ("only 234
German rows, 76 attacks -- thin for a dedicated fit").

WHY THIS SCRIPT, NOT A FULL 4-DETECTOR RESCORE
------------------------------------------------
The roadmap's own text names the starting point precisely: "a calibrated
German-specific threshold on protectai_injection alone (German AUC 0.872
standalone, higher than the 0.819 it contributes diluted inside the pooled
ensemble)". Rescoring the entire fused ensemble (anchors + protectai_injection
+ madhurjindal_jailbreak + toxic_bert) against the full ~13k-row suite to ask
a question that only needs ONE detector, on the ~4,200 German rows, is the
kind of exhaustive-but-unfocused benchmarking pass this project's own
discipline argues against -- it would take hours of CPU-bound transformer
inference to answer a question this script answers in minutes.

METHODOLOGY
-----------
- Reuses cached scores where the row id was already scored in a prior run
  (`_evidence/detector_scores/protectai_injection.jsonl`); only NEWLY added
  German rows are scored, and the cache is updated incrementally, not
  overwritten wholesale like scripts/compare_detectors.py's --refresh does.
- Rows from protectai_injection's declared `trained_on` source
  (deepset/prompt-injections) are excluded, same contamination discipline as
  scripts/compare_detectors.py -- fitting or measuring a threshold on data the
  model was trained on would overstate performance.
- Threshold is fit on a random 50% split of the German population and
  evaluated on the held-out other 50% (simple train/test, not k-fold --
  the question is "does a language-specific cut beat the global one",
  which a single honest split answers; k-fold's benefit is tighter CIs, not
  a different answer, and this suite doesn't yet warrant that decision).
- Compared directly against the DEPLOYED global threshold
  (scripts/compare_detectors.py's own `threshold_at_fpr` on the full,
  language-pooled population from the most recent `_evidence/
  detector_comparison.json`), so the finding is "does language-aware
  thresholding help" rather than an arbitrary number in isolation.

Usage:
    python -m scripts.calibrate_german_threshold
"""
import json
import os
import random

from core.detectors import get_detector
from evaluation.metrics import bootstrap_ci, fmt_ci, recall_at_fpr, roc_auc, threshold_at_fpr

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
COMPARISON_FILE = os.path.join("_evidence", "detector_comparison.json")
REPORT_FILE = os.path.join("_evidence", "german_threshold_calibration.json")
DETECTOR_NAME = "protectai_injection"
SEED = 20260824
FPR_BUDGET = 0.05
N_BOOT = 1000


def load_suite():
    rows = []
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_cache():
    path = os.path.join(SCORES_DIR, f"{DETECTOR_NAME}.jsonl")
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[r["id"]] = r["score"]
    return cache


def save_cache(cache):
    path = os.path.join(SCORES_DIR, f"{DETECTOR_NAME}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rid, score in cache.items():
            f.write(json.dumps({"id": rid, "score": score}) + "\n")


def score_missing(rows, cache, detector, batch_size=16):
    missing = [r for r in rows if r["id"] not in cache]
    if not missing:
        print(f"  all {len(rows)} rows already cached")
        return cache
    print(f"  scoring {len(missing)} new rows (of {len(rows)} needed)...")
    texts = [r["text"] for r in missing]
    scores = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        scores.extend(detector.score_batch(chunk))
        if (i // batch_size) % 20 == 0:
            print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")
    for r, s in zip(missing, scores):
        cache[r["id"]] = s
    save_cache(cache)
    return cache


def load_deployed_threshold():
    """Pulls the currently-deployed protectai_injection threshold from the
    most recent full detector comparison, if one exists -- so this script's
    finding is stated relative to what is actually live, not a number
    invented for this run."""
    if not os.path.exists(COMPARISON_FILE):
        return None
    with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
        report = json.load(f)
    for entry in report.get("results", report if isinstance(report, list) else []):
        if isinstance(entry, dict) and entry.get("detector") == DETECTOR_NAME:
            return entry.get("threshold")
    return None


def main():
    random.seed(SEED)
    rows = load_suite()
    detector = get_detector(DETECTOR_NAME)
    excluded = set(detector.trained_on)

    de_rows = [r for r in rows if r["language"] == "de" and r["source"] not in excluded]
    labels_all = [r["label"] for r in de_rows]
    print(f"German rows (contamination-excluded): {len(de_rows)}  "
          f"attacks={sum(labels_all)}  benign={len(labels_all) - sum(labels_all)}")

    cache = load_cache()
    print(f"[{DETECTOR_NAME}]")
    cache = score_missing(de_rows, cache, detector)

    scores_all = [cache[r["id"]] for r in de_rows]

    # Honest split: fit the threshold on one half, evaluate on the other.
    idx = list(range(len(de_rows)))
    random.shuffle(idx)
    half = len(idx) // 2
    fit_idx, eval_idx = idx[:half], idx[half:]

    fit_scores = [scores_all[i] for i in fit_idx]
    fit_labels = [labels_all[i] for i in fit_idx]
    eval_scores = [scores_all[i] for i in eval_idx]
    eval_labels = [labels_all[i] for i in eval_idx]

    if len(set(fit_labels)) < 2 or len(set(eval_labels)) < 2:
        raise SystemExit("A split ended up single-class -- suite too small or too imbalanced "
                          "for this method. Re-check the suite's German attack/benign balance.")

    german_threshold = threshold_at_fpr(fit_scores, fit_labels, FPR_BUDGET)

    auc_de = bootstrap_ci(eval_scores, eval_labels, roc_auc, n_boot=N_BOOT)
    recall_de_at_german_threshold = sum(
        1 for s, lab in zip(eval_scores, eval_labels) if lab == 1 and s >= german_threshold
    ) / max(1, sum(eval_labels))
    fpr_de_at_german_threshold = sum(
        1 for s, lab in zip(eval_scores, eval_labels) if lab == 0 and s >= german_threshold
    ) / max(1, len(eval_labels) - sum(eval_labels))
    recall_de_at_budget = bootstrap_ci(
        eval_scores, eval_labels, lambda s, lab: recall_at_fpr(s, lab, budget=FPR_BUDGET), n_boot=N_BOOT
    )

    deployed_threshold = load_deployed_threshold()
    print(f"\nGerman-specific threshold (fit on held-in half): {german_threshold:.4f}")
    if deployed_threshold is not None:
        print(f"Deployed (pooled, all-language) threshold:        {deployed_threshold:.4f}")

    print(f"\nHeld-out German half (n={len(eval_idx)}):")
    print(f"  AUC:                      {fmt_ci(auc_de, pct=False)}")
    print(f"  Recall @ {FPR_BUDGET:.0%} FPR budget:   {fmt_ci(recall_de_at_budget)}")
    print(f"  Recall @ German threshold: {recall_de_at_german_threshold:.1%}")
    print(f"  FPR @ German threshold:    {fpr_de_at_german_threshold:.1%}")

    result = None
    if deployed_threshold is not None:
        recall_de_at_deployed = sum(
            1 for s, lab in zip(eval_scores, eval_labels) if lab == 1 and s >= deployed_threshold
        ) / max(1, sum(eval_labels))
        fpr_de_at_deployed = sum(
            1 for s, lab in zip(eval_scores, eval_labels) if lab == 0 and s >= deployed_threshold
        ) / max(1, len(eval_labels) - sum(eval_labels))
        print(f"\n  Recall @ deployed (pooled) threshold: {recall_de_at_deployed:.1%}")
        print(f"  FPR @ deployed (pooled) threshold:    {fpr_de_at_deployed:.1%}")
        delta_recall = recall_de_at_german_threshold - recall_de_at_deployed
        print(f"\n  Recall delta from switching to a German-specific threshold: {delta_recall:+.1%}")
        print(f"  (FPR moves from {fpr_de_at_deployed:.1%} to {fpr_de_at_german_threshold:.1%})")
        result = {
            "recall_at_deployed_threshold": recall_de_at_deployed,
            "fpr_at_deployed_threshold": fpr_de_at_deployed,
            "recall_delta": delta_recall,
        }

    report = {
        "seed": SEED,
        "detector": DETECTOR_NAME,
        "fpr_budget": FPR_BUDGET,
        "n_german_rows": len(de_rows),
        "n_german_attacks": sum(labels_all),
        "n_fit": len(fit_idx),
        "n_eval": len(eval_idx),
        "german_threshold": german_threshold,
        "deployed_threshold": deployed_threshold,
        "auc_on_held_out_german_half": auc_de,
        "recall_at_fpr_budget_held_out": recall_de_at_budget,
        "recall_at_german_threshold": recall_de_at_german_threshold,
        "fpr_at_german_threshold": fpr_de_at_german_threshold,
        "comparison_to_deployed": result,
    }
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
