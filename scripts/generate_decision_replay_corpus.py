"""
Script to generate the frozen decision replay corpus from data/eval_suite.jsonl.

Samples ~300 rows stratified by (attack_class, language) including benign.
Scores each row through core.risk.assess_risk(text) with cache bypassed,
and saves the resulting baseline decisions to tests/fixtures/decision_replay_corpus.jsonl.

Usage:
    python -m scripts.generate_decision_replay_corpus
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import core.risk as risk_mod
from core.risk import assess_risk


CORPUS_DEST = os.path.join("tests", "fixtures", "decision_replay_corpus.jsonl")
SOURCE_SUITE = os.path.join("data", "eval_suite.jsonl")
TARGET_SAMPLE_SIZE = 300
RANDOM_SEED = 42


def sample_stratified_rows(source_path: str = SOURCE_SUITE, target_total: int = TARGET_SAMPLE_SIZE) -> List[Dict[str, Any]]:
    """Deterministically samples target_total rows stratified by (attack_class, language)."""
    with open(source_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        ac = row.get("attack_class", "unknown")
        lang = row.get("language", "unknown")
        strata[(ac, lang)].append(row)

    total_rows = len(rows)
    rng = random.Random(RANDOM_SEED)

    # Calculate proportional allocation with a minimum of 1 per stratum
    allocations: Dict[Tuple[str, str], int] = {}
    for key, group in strata.items():
        prop = len(group) / total_rows * target_total
        allocations[key] = max(1, int(round(prop)))

    # Adjust to match target_total exactly if rounding caused slight deviation
    allocated_total = sum(allocations.values())
    diff = target_total - allocated_total
    if diff != 0:
        largest_key = max(strata.keys(), key=lambda k: len(strata[k]))
        allocations[largest_key] += diff

    sampled_rows: List[Dict[str, Any]] = []
    for key in sorted(strata.keys()):
        group = strata[key]
        n_sample = min(allocations[key], len(group))
        # Deterministic sample from sorted group for 100% reproducibility
        sorted_group = sorted(group, key=lambda r: r["id"])
        chosen = rng.sample(sorted_group, n_sample)
        sampled_rows.extend(chosen)

    # Sort final list by id for deterministic order
    sampled_rows.sort(key=lambda r: r["id"])
    return sampled_rows


def generate_corpus(dest_path: str = CORPUS_DEST, target_total: int = TARGET_SAMPLE_SIZE) -> List[Dict[str, Any]]:
    """Scores sampled rows and writes tests/fixtures/decision_replay_corpus.jsonl."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    sampled = sample_stratified_rows(target_total=target_total)
    print(f"Sampled {len(sampled)} rows across strata. Scoring through core.risk.assess_risk...")

    # Bypass cache so scoring evaluates the real decision engine directly
    orig_lookup = risk_mod.lookup_cache
    orig_save = risk_mod.save_cache_entry
    risk_mod.lookup_cache = lambda prompt, vec: (None, None)
    risk_mod.save_cache_entry = lambda *a, **k: None

    results = []
    t0 = time.perf_counter()

    try:
        for idx, row in enumerate(sampled):
            text = row["text"]
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

            risk_level, details = assess_risk(text)
            source = details.get("source", "unknown")

            corpus_entry = {
                "id": row["id"],
                "text": text,
                "text_sha256": text_sha256,
                "attack_class": row.get("attack_class", "unknown"),
                "language": row.get("language", "unknown"),
                "expected_risk": risk_level,
                "expected_source": source,
            }
            results.append(corpus_entry)

            if (idx + 1) % 50 == 0 or idx + 1 == len(sampled):
                elapsed = time.perf_counter() - t0
                print(f"[{idx + 1}/{len(sampled)}] Scored rows ({elapsed:.1f}s elapsed)...")
    finally:
        risk_mod.lookup_cache = orig_lookup
        risk_mod.save_cache_entry = orig_save

    with open(dest_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_time = time.perf_counter() - t0
    print(f"Successfully generated {len(results)} rows in {dest_path} ({total_time:.1f}s).")
    return results


if __name__ == "__main__":
    generate_corpus()
