import os
import json
import sys
import time
import pandas as pd
from datasets import load_dataset
from core.risk import assess_risk
from core.output_guardrails import assess_output
from core.semantic_judge import judge_available
from tests.benchmark import is_positive_prediction
from api.main import health_check
from tqdm import tqdm

EVIDENCE_DIR = "_evidence"

def check_system_readiness(allow_degraded=False):
    """
    Verifies the system is in a state where measured numbers mean something.

    A degraded health check is FATAL by default: the pipeline fails closed to
    HIGH when the judge is down, so evaluating in that state measures
    infrastructure availability rather than detection quality.
    """
    print("Checking system readiness...")
    status = health_check()
    with open(os.path.join(EVIDENCE_DIR, "system_readiness.json"), "w") as f:
        json.dump(status, f, indent=4)
    print(f"System Readiness: {status['status']}")
    for k, v in status.get('checks', {}).items():
        print(f"  - {k}: {'Pass' if v else 'Fail'}")

    available, detail = judge_available()
    print(f"Judge readiness: {'OK' if available else 'FAILED'} — {detail}")

    if not available and not allow_degraded:
        print(
            "\nABORTING: semantic judge unreachable. Metrics produced now would\n"
            "reflect judge downtime (fail-closed to HIGH), not classifier quality.\n"
            "Start the judge backend, or pass --allow-degraded to override.",
            file=sys.stderr,
        )
        sys.exit(1)

    status["judge_available"] = available
    status["judge_detail"] = detail
    return status

def run_evaluation():
    print("Loading deepset/prompt-injections dataset...")
    # Using the real dataset
    ds = load_dataset('deepset/prompt-injections', split='train')
    
    # Take a representative sample to ensure the evaluation runs in a reasonable time, 
    # but still > 500 prompts as requested previously (we'll use all 546).
    prompts = ds['text']
    labels = ds['label']  # 1 = malicious/injection, 0 = benign

    results = []
    
    tp = fp = tn = fn = 0
    block_count = restrict_count = allow_count = ambiguous_count = 0
    latencies = []

    print("Running evaluation harness...")
    for prompt, label in tqdm(zip(prompts, labels), total=len(prompts)):
        start_time = time.perf_counter()
        
        try:
            risk_level, details = assess_risk(prompt)
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)

            # Classifications.
            # Scoping decisions (domain guardrail) are NOT malice predictions —
            # see tests/benchmark.py for why conflating them invalidated the
            # previous evaluation.
            is_malicious_true = bool(label == 1)
            is_malicious_pred = is_positive_prediction(
                risk_level, details.get("source", "unknown")
            )

            # Detailed counters
            if risk_level == "HIGH": block_count += 1
            elif risk_level == "MEDIUM": restrict_count += 1
            elif risk_level == "LOW": allow_count += 1
            else: ambiguous_count += 1

            # Confusion Matrix
            if is_malicious_true and is_malicious_pred: tp += 1
            elif not is_malicious_true and is_malicious_pred: fp += 1
            elif not is_malicious_true and not is_malicious_pred: tn += 1
            elif is_malicious_true and not is_malicious_pred: fn += 1
            
            # Simulated output guardrail test for a subset (if it was an allow, check output guardrail)
            # We don't have real LLM responses, so we just run the input through the output guardrail
            # to see if it catches anything. This is purely to test the output path.
            out_decision = "N/A"
            if risk_level == "LOW":
                out_dec, out_det = assess_output(prompt)
                out_decision = out_dec

            results.append({
                "prompt": prompt,
                "label": "malicious" if label == 1 else "benign",
                "prediction_risk": risk_level,
                "source": details.get("source", "unknown"),
                "topicality": details.get("topicality", "UNKNOWN"),
                "judge_invoked": details.get("judge_invoked", False),
                "is_correct": is_malicious_pred == is_malicious_true,
                "latency_ms": round(latency, 2),
                "output_guardrail_fallback": out_decision
            })

        except Exception as e:
            print(f"Error evaluating prompt: {e}")

    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    accuracy = (tp + tn) / len(prompts) if len(prompts) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    summary = {
        "total_cases": len(prompts),
        "correct_predictions": tp + tn,
        "incorrect_predictions": fp + fn,
        "accuracy": accuracy,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "confusion_matrix": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        },
        "risk_levels": {
            "HIGH": block_count,
            "MEDIUM": restrict_count,
            "LOW": allow_count,
            "OTHER": ambiguous_count
        },
        "latency_summary_ms": {
            "average": avg_latency,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0
        }
    }

    # Save outputs
    with open(os.path.join(EVIDENCE_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(EVIDENCE_DIR, "evaluation_results.csv"), index=False)

    print("\n=== Evaluation Complete ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"Results saved to {EVIDENCE_DIR}/")

if __name__ == "__main__":
    allow_degraded = "--allow-degraded" in sys.argv
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    check_system_readiness(allow_degraded=allow_degraded)
    run_evaluation()
