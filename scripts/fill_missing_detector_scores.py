"""
Incrementally completes a detector's score cache against the full eval
suite -- scores only rows missing from `_evidence/detector_scores/<name>.jsonl`
and appends, rather than scripts/compare_detectors.py's --refresh, which
rescores EVERY row from scratch the moment even one id is missing from
cache. That distinction is the entire reason this script exists: after
scripts/build_eval_suite.py grows the suite, --refresh would burn hours
re-scoring rows that were already scored and unchanged.

Usage:
    python -m scripts.fill_missing_detector_scores anchors protectai_injection
"""
import json
import os
import sys

from core.detectors import get_detector

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
SCORES_DIR = os.path.join("_evidence", "detector_scores")


def load_suite():
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fill(name, rows, batch_size=16):
    path = os.path.join(SCORES_DIR, f"{name}.jsonl")
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[r["id"]] = r["score"]

    missing = [r for r in rows if r["id"] not in cache]
    print(f"[{name}] {len(cache)} cached, {len(missing)} missing of {len(rows)} total")
    if not missing:
        return

    detector = get_detector(name)
    texts = [r["text"] for r in missing]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        scores = detector.score_batch(chunk)
        for r, s in zip(missing[i:i + batch_size], scores):
            cache[r["id"]] = s
        if (i // batch_size) % 20 == 0:
            print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")
            with open(path, "w", encoding="utf-8") as f:
                for rid, score in cache.items():
                    f.write(json.dumps({"id": rid, "score": score}) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        for rid, score in cache.items():
            f.write(json.dumps({"id": rid, "score": score}) + "\n")
    print(f"  done -> {path} ({len(cache)} total cached)")


def main():
    names = sys.argv[1:]
    if not names:
        raise SystemExit("Usage: python -m scripts.fill_missing_detector_scores <detector> [<detector> ...]")
    rows = load_suite()
    for name in names:
        fill(name, rows)


if __name__ == "__main__":
    main()
