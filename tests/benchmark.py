"""
Canonical evaluation harness for the Gatekeeper risk pipeline.

This is the single source of truth for published metrics.  Two methodological
rules are enforced here, both of which the previous harness violated:

1. SAFETY IS SCORED, SCOPING IS NOT.
   `risk_level` answers "is this an attack?".  `topicality` answers "is this
   about my product's subject area?".  The old harness counted an off-domain
   MEDIUM as a malice prediction, so a benign question about cooking was
   recorded as a false positive.  With a 5-document domain corpus, that alone
   produced FPR ~98% (79 of 80 sampled benign prompts).  Topicality is now
   reported in its own confusion-independent section.

2. AN OFFLINE JUDGE INVALIDATES THE RUN.
   `judge_arbitration` fails closed to HIGH when the judge is unreachable.
   Benchmarking in that state measures sidecar uptime, not classifier quality,
   so the run aborts unless --allow-offline-judge is passed (in which case the
   output is explicitly stamped as methodologically invalid).

Usage:
    python -m tests.benchmark
    python -m tests.benchmark --limit 100
    python -m tests.benchmark --allow-offline-judge   # produces INVALID output
"""
import argparse
import json
import sys
import time
from collections import Counter

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from core.cache import flush_cache
from core.config import settings
from core.logger import get_logger
from core.risk import assess_risk
from core.semantic_judge import judge_available

logger = get_logger(__name__)

RESULTS_FILE = "benchmark_results.json"
ROWS_FILE = "benchmark_rows.csv"

# Sources that represent a SCOPING decision rather than a SAFETY decision.
# A MEDIUM produced by one of these is not a malice claim and must never be
# scored as a positive prediction.
SCOPING_SOURCES = {"domain_guardrail"}


def is_positive_prediction(risk_level: str, source: str, strict: bool = False) -> bool:
    """
    Maps a pipeline verdict to a binary malice prediction.

    strict=False : HIGH or MEDIUM counts as "flagged as malicious"
                   (operational view — anything not cleanly allowed)
    strict=True  : only HIGH counts
                   (conservative view — only confident blocks)
    """
    if source in SCOPING_SOURCES:
        return False
    if strict:
        return risk_level == "HIGH"
    return risk_level in ("HIGH", "MEDIUM")


def compute_metrics(rows, strict=False):
    """Builds a confusion matrix and derived metrics from evaluated rows."""
    tp = fp = tn = fn = 0
    for r in rows:
        pred = is_positive_prediction(r["risk_level"], r["source"], strict=strict)
        truth = r["is_malicious_true"]
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "mode": "strict (HIGH only)" if strict else "operational (HIGH or MEDIUM)",
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall_tpr": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def evaluate(prompts, labels, run_name):
    """Runs the pipeline over the dataset and returns per-row records."""
    rows = []
    latencies = []

    print(f"\n--- {run_name} ---")
    for prompt, label in tqdm(zip(prompts, labels), total=len(prompts), desc=run_name):
        start = time.perf_counter()
        try:
            risk_level, details = assess_risk(prompt)
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            continue
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        rows.append({
            "prompt": prompt,
            "label": "malicious" if label == 1 else "benign",
            "is_malicious_true": bool(label == 1),
            "risk_level": risk_level,
            "source": details.get("source", "unknown"),
            "topicality": details.get("topicality", "UNKNOWN"),
            "judge_invoked": details.get("judge_invoked", False),
            "semantic_score": details.get("semantic_score"),
            "meta_intent_score": details.get("meta_intent_score"),
            "domain_score": details.get("domain_score"),
            "latency_ms": round(latency, 2),
        })

    latencies.sort()

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)] if latencies else 0.0

    return rows, {
        "run_name": run_name,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency_ms": pct(0.50),
        "p95_latency_ms": pct(0.95),
        "p99_latency_ms": pct(0.99),
    }


def summarize(rows, latency_stats):
    """Attaches metrics, decision breakdown, and topicality stats to a run."""
    operational = compute_metrics(rows, strict=False)
    strict = compute_metrics(rows, strict=True)

    benign_rows = [r for r in rows if not r["is_malicious_true"]]
    out_of_domain_benign = sum(1 for r in benign_rows if r["topicality"] == "OUT_OF_DOMAIN")

    return {
        **latency_stats,
        "metrics_operational": operational,
        "metrics_strict": strict,
        "risk_distribution": dict(Counter(r["risk_level"] for r in rows)),
        "source_distribution": dict(Counter(r["source"] for r in rows)),
        "judge_invocations": sum(1 for r in rows if r["judge_invoked"]),
        "topicality": {
            "mode": settings.DOMAIN_GUARDRAIL_MODE,
            "distribution": dict(Counter(r["topicality"] for r in rows)),
            "benign_flagged_out_of_domain": out_of_domain_benign,
            "benign_total": len(benign_rows),
            "note": "Scoping signal. Excluded from the safety confusion matrix by design.",
        },
    }


def print_run(summary):
    op = summary["metrics_operational"]
    st = summary["metrics_strict"]
    cm = op["confusion_matrix"]
    print(f"\nResults for {summary['run_name']}:")
    print(f"  Latency  avg={summary['avg_latency_ms']:.2f}ms  "
          f"p50={summary['p50_latency_ms']:.2f}ms  "
          f"p95={summary['p95_latency_ms']:.2f}ms  "
          f"p99={summary['p99_latency_ms']:.2f}ms")
    print("  [Operational: HIGH or MEDIUM = flagged]")
    print(f"    Accuracy={op['accuracy']:.2%}  Precision={op['precision']:.2%}  "
          f"Recall={op['recall_tpr']:.2%}  F1={op['f1']:.3f}  FPR={op['fpr']:.2%}")
    print(f"    TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}")
    print("  [Strict: HIGH only = flagged]")
    print(f"    Accuracy={st['accuracy']:.2%}  Precision={st['precision']:.2%}  "
          f"Recall={st['recall_tpr']:.2%}  F1={st['f1']:.3f}  FPR={st['fpr']:.2%}")
    print(f"  Judge invoked on {summary['judge_invocations']} prompts")
    print(f"  Decision sources: {summary['source_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Gatekeeper evaluation harness")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N prompts (smoke runs).")
    parser.add_argument("--allow-offline-judge", action="store_true",
                        help="Proceed without a reachable judge. Output is marked INVALID.")
    args = parser.parse_args()

    # ---- METHODOLOGY GATE ----
    available, detail = judge_available()
    print(f"Judge readiness: {'OK' if available else 'FAILED'} — {detail}")
    if not available and not args.allow_offline_judge:
        print(
            "\nABORTING: the semantic judge is unreachable.\n"
            "The pipeline fails closed to HIGH when the judge is down, so any\n"
            "metrics produced now would measure judge availability, not detection\n"
            "quality. Start the judge backend, or re-run with --allow-offline-judge\n"
            "to produce explicitly-invalid output for debugging purposes only.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Domain guardrail mode: {settings.DOMAIN_GUARDRAIL_MODE}")
    print("Loading deepset/prompt-injections dataset...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    prompts, labels = ds["text"], ds["label"]
    if args.limit:
        prompts, labels = prompts[: args.limit], labels[: args.limit]
    print(f"Total prompts loaded: {len(prompts)}")

    flush_cache()
    cold_rows, cold_lat = evaluate(prompts, labels, "Cold Cache (uncached, FAISS)")
    cold = summarize(cold_rows, cold_lat)
    print_run(cold)

    warm_rows, warm_lat = evaluate(prompts, labels, "Warm Cache (LRU + FAISS)")
    warm = summarize(warm_rows, warm_lat)
    print_run(warm)

    speedup = (cold["avg_latency_ms"] / warm["avg_latency_ms"]) if warm["avg_latency_ms"] else 0.0
    print(f"\n=== Cache Speedup: {speedup:.2f}x ===")

    payload = {
        "valid": available,
        "invalid_reason": None if available else f"judge offline: {detail}",
        "config": {
            "domain_guardrail_mode": settings.DOMAIN_GUARDRAIL_MODE,
            "semantic_threshold_high": settings.SEMANTIC_THRESHOLD_HIGH,
            "semantic_threshold_medium": settings.SEMANTIC_THRESHOLD_MEDIUM,
            "meta_intent_threshold": settings.META_INTENT_THRESHOLD,
            "domain_threshold": settings.DOMAIN_THRESHOLD,
            "embedding_model": settings.EMBEDDING_MODEL,
            "judge_model": settings.OLLAMA_MODEL,
        },
        "dataset": {"name": "deepset/prompt-injections", "n": len(prompts)},
        "cold": cold,
        "warm": warm,
        "speedup": speedup,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=4)
    pd.DataFrame(cold_rows).to_csv(ROWS_FILE, index=False)

    print(f"\nSummary -> {RESULTS_FILE}")
    print(f"Per-prompt rows -> {ROWS_FILE}")
    if not available:
        print("\n*** THESE RESULTS ARE METHODOLOGICALLY INVALID (judge was offline). ***")


if __name__ == "__main__":
    main()
