"""
Evaluates the deterministic detector over the multi-source evaluation suite,
sliced by attack class, language and source, with bootstrap confidence
intervals on every headline number.

This is the script that produces the results chapter. It answers the questions a
single pooled accuracy figure cannot:

  - Which attack class is the detector actually good at?
  - How much of the error is the English-only encoder?
  - Does performance hold across sources, or is it fitted to one distribution?

The semantic judge is excluded (non-deterministic, expensive, not reproducible);
this measures the deterministic signal path only, which is what the thresholds
are calibrated against.

Usage:
    python -m scripts.evaluate_suite
    python -m scripts.evaluate_suite --limit 500 --bootstrap 500
"""
import argparse
import json
import os

from evaluation.metrics import (
    evaluate_by,
    evaluate_slice,
    fmt_ci,
    threshold_at_fpr,
)

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SIGNALS_FILE = os.path.join("_evidence", "suite_signals.jsonl")
REPORT_FILE = os.path.join("_evidence", "suite_evaluation.json")


def load_suite(limit=None):
    if not os.path.exists(SUITE_FILE):
        raise SystemExit(
            f"{SUITE_FILE} not found. Build it first:\n"
            f"    python -m scripts.build_eval_suite"
        )
    records = []
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records[:limit] if limit else records


def extract_signals(records, refresh=False):
    """
    Scores every row with the deterministic detector. Cached — this is the only
    expensive step (one embedding per row).
    """
    if os.path.exists(SIGNALS_FILE) and not refresh:
        cached = {}
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cached[row["id"]] = row
        if all(r["id"] in cached for r in records):
            print(f"Loaded cached signals for {len(records)} rows "
                  f"({SIGNALS_FILE}; --refresh to recompute)")
            return [cached[r["id"]] for r in records]
        print("Cache incomplete — recomputing.")

    from tqdm import tqdm
    from core.embeddings import get_embedding
    from core.risk import _ensure_faiss_initialized, check_meta_intent, hard_ban_triggered
    from core.updates import check_dynamic_threats
    from core.vector_store import threat_store

    _ensure_faiss_initialized()

    scored = []
    print(f"Scoring {len(records)} prompts...")
    for r in tqdm(records):
        symbolic, _ = hard_ban_triggered(r["text"])
        vec = get_embedding(r["text"])
        threat = float(threat_store.get_max_similarity(vec))
        dynamic = float(check_dynamic_threats(vec))
        meta = float(check_meta_intent(vec))
        # Single continuous detector score. Symbolic hits are a deterministic
        # veto and take the maximum by construction.
        score = 1.0 if symbolic else max(threat, dynamic, meta)
        scored.append({**r, "symbolic": bool(symbolic), "threat_score": threat,
                       "dynamic_threat_score": dynamic, "meta_intent_score": meta,
                       "score": score})

    os.makedirs("_evidence", exist_ok=True)
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Cached signals -> {SIGNALS_FILE}")
    return scored


def print_table(title, results):
    print(f"\n--- {title} ---")
    print(f"  {'slice':<22} {'n':>6} {'atk':>6} {'AUC':>22} {'Recall@5%FPR':>22}")
    for name, res in results.items():
        flag = " (!)" if res.get("underpowered") else ""
        if not res.get("evaluable"):
            print(f"  {name[:22]:<22} {res['n']:>6} {res['n_attack']:>6} "
                  f"{'single-class':>22} {'--':>22}{flag}")
            continue
        print(f"  {name[:22]:<22} {res['n']:>6} {res['n_attack']:>6} "
              f"{fmt_ci(res['auc'], pct=False):>22} "
              f"{fmt_ci(res['recall_at_fpr']):>22}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate over the multi-source suite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fpr-budget", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    records = load_suite(limit=args.limit)
    scored = extract_signals(records, refresh=args.refresh)

    scores = [r["score"] for r in scored]
    labels = [r["label"] for r in scored]

    overall = evaluate_slice(scores, labels, fpr_budget=args.fpr_budget,
                             n_boot=args.bootstrap)
    # One global operating point, applied to every slice, so slices are
    # comparable. Per-slice thresholds would let each slice pick its own
    # favourable cutoff and would not reflect a deployable configuration.
    global_threshold = threshold_at_fpr(scores, labels, args.fpr_budget)

    print("\n=== OVERALL (pooled) ===")
    print(f"  n={overall['n']}  attacks={overall['n_attack']}  benign={overall['n_benign']}")
    print(f"  ROC AUC          : {fmt_ci(overall['auc'], pct=False)}")
    print(f"  Recall @ {args.fpr_budget:.0%} FPR  : {fmt_ci(overall['recall_at_fpr'])}")
    print(f"  Operating point  : score >= {global_threshold:.4f}")
    print(f"  Confusion        : {overall.get('confusion_matrix')}")
    print(f"  Precision {overall.get('precision', 0):.1%}  "
          f"Recall {overall.get('recall', 0):.1%}  "
          f"FPR {overall.get('fpr', 0):.1%}  F1 {overall.get('f1', 0):.3f}")

    by_class = evaluate_by(scored, "score", "label", "attack_class",
                           fpr_budget=args.fpr_budget, n_boot=args.bootstrap,
                           threshold=global_threshold)
    by_lang = evaluate_by(scored, "score", "label", "language",
                          fpr_budget=args.fpr_budget, n_boot=args.bootstrap,
                          threshold=global_threshold)
    by_source = evaluate_by(scored, "score", "label", "source",
                            fpr_budget=args.fpr_budget, n_boot=args.bootstrap,
                            threshold=global_threshold)

    print_table("BY ATTACK CLASS", by_class)
    print_table("BY LANGUAGE", by_lang)
    print_table("BY SOURCE", by_source)
    print("\n  (!) = underpowered slice; interval too wide to act on")

    # Attack-only recall per class at the shared operating point. Attack classes
    # are single-class slices (no benign rows of their own), so AUC is undefined
    # for them and recall at the global threshold is the meaningful number.
    print("\n--- ATTACK DETECTION RATE @ global operating point ---")
    for cls in ("prompt_injection", "jailbreak", "harmful_content"):
        rows = [r for r in scored if r["attack_class"] == cls]
        if not rows:
            continue
        hit = sum(1 for r in rows if r["score"] >= global_threshold)
        print(f"  {cls:<20} {hit:>5}/{len(rows):<5} = {hit / len(rows):6.1%}")

    benign = [r for r in scored if r["label"] == 0]
    fp = sum(1 for r in benign if r["score"] >= global_threshold)
    print(f"  {'benign (FPR)':<20} {fp:>5}/{len(benign):<5} = {fp / len(benign):6.1%}")

    report = {
        "config": {
            "fpr_budget": args.fpr_budget,
            "bootstrap_resamples": args.bootstrap,
            "global_threshold": global_threshold,
            "judge_excluded": True,
            "detector": "deterministic signals (anchors + meta-intent + symbolic)",
        },
        "overall": overall,
        "by_attack_class": by_class,
        "by_language": by_lang,
        "by_source": by_source,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
