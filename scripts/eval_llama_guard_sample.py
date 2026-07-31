"""
Evaluates Llama Guard on a STRATIFIED SAMPLE, and re-runs the detector
comparison and the learned-fusion ensemble on exactly those same rows.

WHY A SAMPLE
------------
Llama Guard 3 1B is a generative model on CPU. Its prompt embeds all 13 hazard
category descriptions (~700 tokens of fixed overhead before the user's text),
so a single forward pass is expensive. Full-suite scoring measured at ~5 min per
16-row batch => ~33-40 HOURS for 6,933 rows. That is not a viable measurement on
this hardware, and running it anyway would be the same "measure the wrong thing
because it was easy" mistake this project spent its effort correcting.

A stratified sample gives a defensible AUC with a confidence interval in
minutes. The interval will be WIDER than the full-suite numbers for the other
detectors, and every table this produces says so — a wide honest interval beats
a narrow one that took two days you did not have.

FAIRNESS
--------
Every detector is evaluated on the IDENTICAL sampled rows, read from the same
caches used by scripts.compare_detectors. Llama Guard is scored live; the rest
are looked up. No detector gets a different denominator.

SAMPLE DESIGN
-------------
- ALL harmful_content attack rows. This is the class the whole exercise is
  about, and it is small enough (~250) to take whole.
- A capped random sample of the other attack classes, for comparison context.
- A benign sample sized to keep the attack:benign ratio near the full suite's,
  so the FPR budget and AUC remain comparable in meaning.

Sampling is deterministic (fixed seed) so the run is reproducible and resumable.

Usage:
    python -m scripts.eval_llama_guard_sample
    python -m scripts.eval_llama_guard_sample --per-other-class 150 --benign 400
"""
import argparse
import json
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.detectors import get_detector, get_registry
from evaluation.metrics import (
    bootstrap_ci,
    fmt_ci,
    recall_at_fpr,
    roc_auc,
    threshold_at_fpr,
)

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
SAMPLE_CACHE = os.path.join("_evidence", "detector_scores", "llama_guard_3_1b.sample.jsonl")
REPORT_FILE = os.path.join("_evidence", "llama_guard_sample_report.json")
SEED = 20260724
FUSION_SEED = 20260723


def load_suite():
    rows = []
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_cached(name):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def build_sample(rows, per_other_class, benign_n):
    """Deterministic stratified sample; see module docstring for rationale."""
    rng = random.Random(SEED)
    harmful = [r for r in rows if r["attack_class"] == "harmful_content" and r["label"] == 1]
    injection = [r for r in rows if r["attack_class"] == "prompt_injection" and r["label"] == 1]
    jailbreak = [r for r in rows if r["attack_class"] == "jailbreak" and r["label"] == 1]
    benign = [r for r in rows if r["label"] == 0]

    def take(pool, n):
        return rng.sample(pool, min(n, len(pool)))

    sample = (harmful
              + take(injection, per_other_class)
              + take(jailbreak, per_other_class)
              + take(benign, benign_n))
    rng.shuffle(sample)
    return sample, {
        "harmful_content": len(harmful),
        "prompt_injection": min(per_other_class, len(injection)),
        "jailbreak": min(per_other_class, len(jailbreak)),
        "benign": min(benign_n, len(benign)),
    }


def score_llama_guard(sample, refresh=False):
    """Scores Llama Guard on the sample, caching so a rerun is instant."""
    detector = get_detector("llama_guard_3_1b")
    ok, detail = detector.available()
    if not ok:
        raise SystemExit(
            f"\nllama_guard_3_1b unavailable: {detail}\n"
            f"If this is a memory error, close other applications — the model "
            f"needs ~2.5GB free — and rerun. Nothing else is blocking."
        )

    cached = {}
    if os.path.exists(SAMPLE_CACHE) and not refresh:
        with open(SAMPLE_CACHE, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cached[r["id"]] = r["score"]

    todo = [r for r in sample if r["id"] not in cached]
    if todo:
        from tqdm import tqdm
        print(f"Scoring {len(todo)} rows with Llama Guard "
              f"({len(cached)} already cached)...")
        batch = 8
        with open(SAMPLE_CACHE, "a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(todo), batch)):
                chunk = todo[i:i + batch]
                scores = detector.score_batch([r["text"] for r in chunk])
                for r, s in zip(chunk, scores):
                    cached[r["id"]] = s
                    f.write(json.dumps({"id": r["id"], "score": s}) + "\n")
                    f.flush()  # survive an interrupted run row-by-row
    else:
        print(f"All {len(sample)} sampled rows already cached for Llama Guard.")
    return cached


def evaluate_single(name, scores, labels, fpr_budget, n_boot):
    auc = bootstrap_ci(scores, labels, roc_auc, n_boot=n_boot)
    rec = bootstrap_ci(scores, labels,
                       lambda s, lab: recall_at_fpr(s, lab, budget=fpr_budget),
                       n_boot=n_boot)
    return {"detector": name, "auc": auc, "recall_at_fpr": rec}


def per_class_detection(rows, scores, threshold):
    out = {}
    for cls in ("prompt_injection", "jailbreak", "harmful_content"):
        sub = [s for r, s in zip(rows, scores) if r["attack_class"] == cls]
        if sub:
            out[cls] = sum(1 for s in sub if s >= threshold) / len(sub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-other-class", type=int, default=150)
    ap.add_argument("--benign", type=int, default=400)
    ap.add_argument("--fpr-budget", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rows = load_suite()
    sample, counts = build_sample(rows, args.per_other_class, args.benign)
    n_pos = sum(r["label"] for r in sample)
    print(f"Stratified sample: {len(sample)} rows "
          f"({n_pos} attack / {len(sample) - n_pos} benign)")
    print(f"  composition: {counts}\n")

    # 1. Llama Guard live, everything else from cache — same rows for all.
    lg_scores = score_llama_guard(sample, refresh=args.refresh)

    registry = get_registry()
    detectors = {}  # name -> {id: score} restricted to the sample
    detectors["llama_guard_3_1b"] = lg_scores
    for name in registry:
        if name == "llama_guard_3_1b":
            continue
        cached = load_cached(name)
        if cached and all(r["id"] in cached for r in sample):
            detectors[name] = {r["id"]: cached[r["id"]] for r in sample}

    labels = [r["label"] for r in sample]

    # 2. Per-detector metrics on the sample.
    print(f"\n{'=' * 88}")
    print(f"LLAMA GUARD SAMPLE EVALUATION  ({len(sample)} rows, "
          f"{args.bootstrap} bootstrap, FPR {args.fpr_budget:.0%})")
    print(f"Intervals are WIDER than the full-suite tables — this is a sample.")
    print("=" * 88)
    print(f"{'detector':<24} {'AUC':>22} {'Recall@budget':>22}")

    single_results = []
    for name, sc in detectors.items():
        scores = [sc[r["id"]] for r in sample]
        # Exclude a detector's own training source (parity with full comparison).
        trained_on = set(registry[name].trained_on)
        keep = [(s, r["label"]) for s, r in zip(scores, sample)
                if r["source"] not in trained_on]
        ks, kl = [s for s, _ in keep], [l for _, l in keep]
        if len(set(kl)) < 2:
            continue
        res = evaluate_single(name, ks, kl, args.fpr_budget, args.bootstrap)
        res["n"] = len(kl)
        single_results.append(res)

    for r in sorted(single_results, key=lambda x: -x["auc"]["point"]):
        print(f"{r['detector']:<24} {fmt_ci(r['auc'], pct=False):>22} "
              f"{fmt_ci(r['recall_at_fpr']):>22}")

    # 3. Per-class detection, focused on harmful_content — the point of this.
    print(f"\n{'-' * 88}")
    print("HARMFUL-CONTENT DETECTION at each detector's own budget threshold")
    print(f"(sample has all {counts['harmful_content']} harmful_content rows)")
    hc_rows = [r for r in sample if r["attack_class"] == "harmful_content"]
    hc_table = []
    for name, sc in detectors.items():
        scores = [sc[r["id"]] for r in sample]
        thr = threshold_at_fpr(scores, labels, args.fpr_budget)
        hc_scores = [sc[r["id"]] for r in hc_rows]
        rate = sum(1 for s in hc_scores if s >= thr) / len(hc_scores)
        hc_table.append((name, rate))
    for name, rate in sorted(hc_table, key=lambda x: -x[1]):
        print(f"  {name:<24} {rate:>6.1%}")

    # 4. Learned fusion on the sample, INCLUDING Llama Guard, out-of-fold.
    #    Uncontaminated detectors only, matching the main ensemble analysis.
    fusion_names = [n for n in detectors
                    if not registry[n].trained_on]
    X = np.array([[detectors[n][r["id"]] for n in fusion_names] for r in sample])
    y = np.array(labels)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=FUSION_SEED)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    fusion_auc = bootstrap_ci(list(oof), labels, roc_auc, n_boot=args.bootstrap)
    fusion_thr = threshold_at_fpr(list(oof), labels, args.fpr_budget)
    fusion_hc = sum(1 for r in hc_rows if oof[sample.index(r)] >= fusion_thr) / len(hc_rows)

    # Fusion WITHOUT Llama Guard, same rows, to isolate its marginal value.
    names_no_lg = [n for n in fusion_names if n != "llama_guard_3_1b"]
    X2 = np.array([[detectors[n][r["id"]] for n in names_no_lg] for r in sample])
    oof2 = cross_val_predict(model, X2, y, cv=cv, method="predict_proba")[:, 1]
    fusion_auc_no_lg = bootstrap_ci(list(oof2), labels, roc_auc, n_boot=args.bootstrap)
    fusion_thr2 = threshold_at_fpr(list(oof2), labels, args.fpr_budget)
    fusion_hc_no_lg = sum(1 for r in hc_rows if oof2[sample.index(r)] >= fusion_thr2) / len(hc_rows)

    model.fit(X, y)
    coefs = dict(sorted(
        zip(fusion_names, model.named_steps["logisticregression"].coef_[0].tolist()),
        key=lambda kv: -abs(kv[1])))

    print(f"\n{'-' * 88}")
    print("DOES LLAMA GUARD IMPROVE THE FUSION? (out-of-fold, same rows)")
    print(f"  fusion WITHOUT llama guard : AUC {fmt_ci(fusion_auc_no_lg, pct=False)}"
          f"  harmful {fusion_hc_no_lg:.1%}")
    print(f"  fusion WITH    llama guard : AUC {fmt_ci(fusion_auc, pct=False)}"
          f"  harmful {fusion_hc:.1%}")
    print(f"\n  fusion coefficients (with llama guard):")
    for name, c in coefs.items():
        print(f"    {name:<24} {c:+.4f}")

    payload = {
        "sample": {"n": len(sample), "composition": counts,
                   "n_attack": int(n_pos), "seed": SEED},
        "config": {"fpr_budget": args.fpr_budget, "bootstrap": args.bootstrap},
        "single_detectors": single_results,
        "harmful_content_detection": dict(hc_table),
        "fusion_with_llama_guard": {
            "auc": fusion_auc, "harmful_content_detected": fusion_hc,
            "coefficients": coefs},
        "fusion_without_llama_guard": {
            "auc": fusion_auc_no_lg, "harmful_content_detected": fusion_hc_no_lg},
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
