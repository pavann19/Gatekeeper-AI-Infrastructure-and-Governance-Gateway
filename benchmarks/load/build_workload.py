"""
Builds the fixed load-test workload from this project's own eval suite.

    PYTHONPATH=. python benchmarks/load/build_workload.py

Deterministic: fixed seed, fixed sample size, stratified by (label,
language) proportional to the full 13,011-row suite so the fixed benign/
attack/German mix isn't an arbitrary guess -- it mirrors the corpus the
project already calibrates against, just capped to a size a closed-loop
load test can cycle through many times in one run. Re-running this script
regenerates byte-identical output; the committed workload_v1.jsonl and its
.sha256 sidecar are checked into git specifically so nobody has to.

Writes:
  benchmarks/load/workload_v1.jsonl        N_TOTAL rows, one JSON object each
  benchmarks/load/workload_v1.jsonl.sha256 the SHA-256 the bench harness
                                            verifies before every run
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SUITE = os.path.join(REPO, "data", "eval_suite.jsonl")
OUT = os.path.join(HERE, "workload_v1.jsonl")

N_TOTAL = 300
SEED = 20260916  # fixed: the date this workload was cut, not "today"


def main() -> None:
    rows = [json.loads(line) for line in open(SUITE, encoding="utf-8")]
    by_stratum: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_stratum[(r["label"], r.get("language", "?"))].append(r)

    total = len(rows)
    rng = random.Random(SEED)
    strata = sorted(by_stratum, key=lambda k: (-len(by_stratum[k]), k))

    picked: list[dict] = []
    for key in strata:
        pool = by_stratum[key][:]
        rng.shuffle(pool)
        take = max(1, round(N_TOTAL * len(by_stratum[key]) / total))
        picked.extend(pool[:take])

    rng.shuffle(picked)
    picked = picked[:N_TOTAL]
    # Deterministic final order: shuffling twice with a fixed seed is still
    # deterministic, but sort by id afterwards so the file's row order is
    # stable under any future refactor of the sampling loop above, not just
    # under an unchanged RNG call sequence.
    picked.sort(key=lambda r: r["id"])

    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(picked):
            entry = {
                "workload_index": i,
                "id": r["id"],
                "text": r["text"],
                "label": r["label"],
                "attack_class": r.get("attack_class"),
                "language": r.get("language"),
                "source": r.get("source"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    with open(OUT + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(OUT)}\n")

    lang = defaultdict(int)
    label = defaultdict(int)
    for r in picked:
        lang[r.get("language", "?")] += 1
        label[r["label"]] += 1
    print(f"wrote {len(picked)} rows -> {OUT}")
    print(f"  language mix: {dict(lang)}")
    print(f"  label mix (0=benign,1=attack): {dict(label)}")
    print(f"  sha256: {digest}")


if __name__ == "__main__":
    main()
