"""
Measures the real behavioral impact of the is_educational wiring fix
(docs/ENGINEERING_ASSESSMENT.md section 1z) on the full 6,933-row eval
suite, WITHOUT requiring a live judge.

WHY THIS IS SAFE WITHOUT THE JUDGE
------------------------------------
`is_educational` only matters in fuse_signals' ambiguous zone
(threshold_medium <= score < threshold_high), and in that zone it changes
`fusion_judge_pending` (risk_level MEDIUM, judge_required=True) into
`fusion_educational_safe_harbor` (risk_level MEDIUM, judge_required=False)
-- the RISK LEVEL is identical either way; the only difference is whether
judge arbitration is skipped. Before this fix, `is_educational` was always
False (a dead function), so the safe-harbor branch could never fire at
all -- every ambiguous-zone row unconditionally went to
`fusion_judge_pending`. That means the full behavioral delta is exactly:
how many ambiguous-zone rows now have is_educational=True?

This script computes that directly from cached detector scores (fast) plus
a fresh embedding pass for check_educational_context (the only new
computation this fix requires), with no judge/Ollama dependency.

Usage:
    python -m scripts.analyze_educational_safe_harbor_impact
"""
import json
import os

import core.fusion as fusion_mod
from core.embeddings import get_embedding
from core.risk import _ensure_faiss_initialized, check_educational_context

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
LIVE_FEATURES = ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"]


def load_scores(name):
    out = {}
    with open(os.path.join(SCORES_DIR, f"{name}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["score"]
    return out


def main():
    fusion_mod.policy_available()
    policy = fusion_mod._policy
    t_high, t_med = policy["threshold_high"], policy["threshold_medium"]
    print(f"threshold_high={t_high:.4f}  threshold_medium={t_med:.4f}")

    rows = [json.loads(l) for l in open(SUITE_FILE, encoding="utf-8")]
    caches = {n: load_scores(n) for n in LIVE_FEATURES}
    usable = [r for r in rows if all(r["id"] in caches[n] for n in LIVE_FEATURES)]

    _ensure_faiss_initialized()

    ambiguous_rows = []
    for r in usable:
        feature_values = {n: caches[n][r["id"]] for n in LIVE_FEATURES}
        score = fusion_mod._apply_policy(feature_values)
        if t_med <= score < t_high:
            ambiguous_rows.append((r, score))

    print(f"\nAmbiguous-zone rows (would have called the judge before this fix): "
          f"{len(ambiguous_rows)} / {len(usable)}")

    flipped = []
    print("Computing is_educational for each ambiguous-zone row "
          f"({len(ambiguous_rows)} embeddings)...")
    for i, (r, score) in enumerate(ambiguous_rows):
        vec = get_embedding(r["text"])
        if check_educational_context(vec):
            flipped.append((r, score))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ambiguous_rows)}")

    print(f"\nRows that now skip the judge (fusion_judge_pending -> "
          f"fusion_educational_safe_harbor): {len(flipped)}")
    print("Both branches return risk_level=MEDIUM -- this count is the ENTIRE "
          "behavioral delta of the fix, not a change in any HIGH decision.")

    from collections import Counter
    print("\nBy attack_class among flipped rows:")
    print(Counter(r["attack_class"] for r, _ in flipped))
    print("\nSample flipped rows:")
    for r, score in flipped[:10]:
        print(f"  [{r['attack_class']:<16} score={score:.3f}] {r['text'][:90]}")

    report = {
        "threshold_high": t_high, "threshold_medium": t_med,
        "n_total": len(usable), "n_ambiguous_zone": len(ambiguous_rows),
        "n_flipped_to_safe_harbor": len(flipped),
        "flipped_by_class": dict(Counter(r["attack_class"] for r, _ in flipped)),
        "flipped_rows": [{"id": r["id"], "attack_class": r["attack_class"],
                          "label": r["label"], "score": score, "text": r["text"]}
                         for r, score in flipped],
    }
    out_path = os.path.join("_evidence", "educational_safe_harbor_impact.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
