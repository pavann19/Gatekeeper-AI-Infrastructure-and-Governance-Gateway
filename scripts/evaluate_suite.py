"""
Evaluates the risk detector over the multi-source evaluation suite, sliced by
attack class, language and source, with bootstrap confidence intervals on
every headline number.

This is the script that produces the results chapter. It answers the questions a
single pooled accuracy figure cannot:

  - Which attack class is the detector actually good at?
  - How much of the error is the English-only encoder?
  - Does performance hold across sources, or is it fitted to one distribution?

Two scoring modes, both judge-excluded (the judge is non-deterministic,
expensive, and not reproducible):

  deterministic (default) — anchors + meta-intent + symbolic veto only. This
      is the fast-path floor the thresholds in core/config.py are calibrated
      against. Cheap: one embedding per row.

  --fusion — the full core.fusion.fused_threat_score() ensemble as deployed
      (the richest reachable upgrade tier), falling back to the deterministic
      score only on rows where fusion reports itself unavailable. This is the
      accuracy of the system people actually run, minus the judge. Expensive:
      ~0.5 s per row for the 8-detector ensemble, so use --jobs and/or
      --limit. Writes to its own signals cache and report file; never
      touches the deterministic artifacts.

Usage:
    python -m scripts.evaluate_suite
    python -m scripts.evaluate_suite --limit 500 --bootstrap 500
    python -m scripts.evaluate_suite --fusion --jobs 3
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

_SIGNALS_FILE = {
    "deterministic": os.path.join("_evidence", "suite_signals.jsonl"),
    "fusion": os.path.join("_evidence", "suite_signals_fusion.jsonl"),
}
_REPORT_FILE = {
    "deterministic": os.path.join("_evidence", "suite_evaluation.json"),
    "fusion": os.path.join("_evidence", "suite_evaluation_fusion.json"),
}
_DETECTOR_DESC = {
    "deterministic": "deterministic signals (anchors + meta-intent + symbolic)",
    "fusion": "fusion ensemble (fused_threat_score, richest reachable tier; "
              "deterministic fallback on unavailable rows; judge excluded)",
}

# Back-compat aliases — some callers import these by name.
SIGNALS_FILE = _SIGNALS_FILE["deterministic"]
REPORT_FILE = _REPORT_FILE["deterministic"]


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


def _score_row_deterministic(r, deps):
    get_embedding, check_meta_intent, hard_ban_triggered, threat_store = deps
    symbolic, _ = hard_ban_triggered(r["text"])
    vec = get_embedding(r["text"])
    threat = float(threat_store.get_max_similarity(vec))
    meta = float(check_meta_intent(vec))
    # Single continuous detector score. Symbolic hits are a deterministic
    # veto and take the maximum by construction.
    score = 1.0 if symbolic else max(threat, meta)
    return {**r, "symbolic": bool(symbolic), "threat_score": threat,
            "meta_intent_score": meta, "score": score}


def _score_row_fusion(r, deps):
    get_embedding, check_meta_intent, hard_ban_triggered, threat_store = deps
    from core.fusion import fused_threat_score

    symbolic, _ = hard_ban_triggered(r["text"])
    vec = get_embedding(r["text"])
    threat = float(threat_store.get_max_similarity(vec))
    meta = float(check_meta_intent(vec))
    det_floor = max(threat, meta)

    if symbolic:
        return {**r, "symbolic": True, "threat_score": threat,
                "meta_intent_score": meta, "fusion_available": False,
                "fusion_score": None, "fusion_detail": "symbolic veto (fusion skipped)",
                "detector_scores": {},
                "score": 1.0}

    fusion = fused_threat_score(r["text"], anchor_score=threat)
    if fusion.get("available") and fusion.get("score") is not None:
        score = float(fusion["score"])
    else:
        score = det_floor  # honest fallback, not an imputed zero
    return {**r, "symbolic": False, "threat_score": threat,
            "meta_intent_score": meta,
            "fusion_available": bool(fusion.get("available")),
            "fusion_score": fusion.get("score"),
            "fusion_detail": fusion.get("detail"),
            "detector_scores": fusion.get("detector_scores", {}),
            "score": score}


def extract_signals(records, refresh=False, mode="deterministic", jobs=1, refresh_only_features=False):
    """
    Scores every row and caches the result. `mode` selects the scorer:
    'deterministic' (anchors+meta+symbolic, one embedding per row) or 'fusion'
    (the full ensemble, ~0.5s per row). Each mode has its own cache file so
    the two never collide.
    """
    signals_file = _SIGNALS_FILE[mode]
    if mode == "fusion" and refresh_only_features:
        cached = {}
        if os.path.exists(signals_file):
            with open(signals_file, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    cached[row["id"]] = row
        det_signals_file = _SIGNALS_FILE["deterministic"]
        if os.path.exists(det_signals_file):
            with open(det_signals_file, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row["id"] not in cached or not cached[row["id"]].get("threat_score"):
                        cached[row["id"]] = {**cached.get(row["id"], {}), **row}
        import core.fusion as fusion_mod
        fusion_mod._load_policy()
        tier = fusion_mod._policy.get("upgrade_tiers", [fusion_mod._policy])[0] if fusion_mod._policy else {}
        feature_names = [feat for feat in tier.get("feature_order", []) if feat != "anchors"]
        detector_caches = {}
        for feat in feature_names:
            p = os.path.join("_evidence", "detector_scores", f"{feat}.jsonl")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as df:
                    detector_caches[feat] = {json.loads(line)["id"]: json.loads(line)["score"] for line in df}
        updated = []
        for r in records:
            row = cached.get(r["id"], {**r})
            if row.get("symbolic"):
                row["detector_scores"] = {}
                row["fusion_available"] = False
                row["fusion_score"] = None
                row["fusion_detail"] = "symbolic veto (fusion skipped)"
                row["score"] = 1.0
            else:
                det_scores = {"anchors": row.get("threat_score", 0.0)}
                for feat in feature_names:
                    if feat in detector_caches and r["id"] in detector_caches[feat]:
                        det_scores[feat] = detector_caches[feat][r["id"]]
                row["detector_scores"] = det_scores
                verdict = fusion_mod._select_per_class_verdict(det_scores, policy_tier=tier)
                if verdict is not None:
                    winner, class_results = verdict
                    w = class_results[winner]
                    row["fusion_available"] = True
                    row["fusion_score"] = w["score"]
                    row["fusion_detail"] = f"fusion applied (per-class, triggered by {winner}, tier={tier.get('tier_id', '?')})"
                    row["score"] = w["score"]
            updated.append(row)
        with open(signals_file, "w", encoding="utf-8") as f:
            for row in updated:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Rehydrated detector_scores into {signals_file} ({len(updated)} rows)")
        return updated

    if os.path.exists(signals_file) and not refresh:
        cached = {}
        with open(signals_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cached[row["id"]] = row
        if all(r["id"] in cached for r in records):
            print(f"Loaded cached {mode} signals for {len(records)} rows "
                  f"({signals_file}; --refresh to recompute)")
            return [cached[r["id"]] for r in records]
        print(f"{mode} cache incomplete — recomputing.")

    from tqdm import tqdm
    from core.config import settings
    from core.embeddings import get_embedding
    from core.risk import _ensure_faiss_initialized, check_meta_intent, hard_ban_triggered
    from core.vector_store import threat_store

    _ensure_faiss_initialized()
    deps = (get_embedding, check_meta_intent, hard_ban_triggered, threat_store)
    score_row = _score_row_fusion if mode == "fusion" else _score_row_deterministic

    # Warm the ensemble once, single-threaded, before any fan-out: transformers'
    # lazy module loader races when several threads first import a cold model
    # (see core/fusion.py::_warm_detectors). Cheap insurance.
    if mode == "fusion" and records:
        score_row(records[0], deps)

    n_fusion_unavailable = 0
    scored = []
    print(f"Scoring {len(records)} prompts ({mode}, jobs={jobs})...")

    if jobs > 1:
        # Row-level threads replace fusion's own detector-level threads, so
        # turn the latter off to avoid jobs * detectors thread oversubscription.
        orig_parallel = settings.FUSION_PARALLEL
        settings.FUSION_PARALLEL = False
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                for row in tqdm(ex.map(lambda r: score_row(r, deps), records),
                                total=len(records)):
                    scored.append(row)
                    n_fusion_unavailable += (mode == "fusion"
                                             and not row.get("fusion_available")
                                             and not row.get("symbolic"))
        finally:
            settings.FUSION_PARALLEL = orig_parallel
    else:
        for r in tqdm(records):
            row = score_row(r, deps)
            scored.append(row)
            n_fusion_unavailable += (mode == "fusion"
                                     and not row.get("fusion_available")
                                     and not row.get("symbolic"))

    if mode == "fusion" and n_fusion_unavailable:
        print(f"  WARNING: fusion unavailable on {n_fusion_unavailable}/"
              f"{len(scored)} rows — deterministic fallback used for those.")

    os.makedirs("_evidence", exist_ok=True)
    with open(signals_file, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Cached {mode} signals -> {signals_file}")
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
    parser.add_argument("--refresh-only-features", action="store_true",
                        help="fast path: rehydrate detector_scores from cached "
                             "per-detector files without re-running model inference")
    parser.add_argument("--fusion", action="store_true",
                        help="score with the full fusion ensemble instead of "
                             "the deterministic floor (slow; ~0.5s/row)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="row-level worker threads for --fusion "
                             "(3-4 is a sane max on a laptop)")
    args = parser.parse_args()

    mode = "fusion" if args.fusion else "deterministic"
    report_file = _REPORT_FILE[mode]

    records = load_suite(limit=args.limit)
    scored = extract_signals(
        records, refresh=args.refresh, mode=mode, jobs=args.jobs,
        refresh_only_features=args.refresh_only_features
    )

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

    fusion_unavail = sum(1 for r in scored
                         if not r.get("symbolic") and r.get("fusion_available") is False)
    report = {
        "config": {
            "mode": mode,
            "fpr_budget": args.fpr_budget,
            "bootstrap_resamples": args.bootstrap,
            "global_threshold": global_threshold,
            "judge_excluded": True,
            "detector": _DETECTOR_DESC[mode],
            "n_rows": len(scored),
            "n_fusion_unavailable": fusion_unavail if mode == "fusion" else None,
        },
        "overall": overall,
        "by_attack_class": by_class,
        "by_language": by_lang,
        "by_source": by_source,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {report_file}")


if __name__ == "__main__":
    main()
