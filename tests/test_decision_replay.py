"""
Decision-Replay Regression Gate.

Re-runs a frozen stratified sample of ~300 rows from data/eval_suite.jsonl through
core.risk.assess_risk(text) and asserts that risk_level matches expected_risk exactly
with zero tolerance.

The frozen baseline used unavailable judges. Canonical replay explicitly pins its
model tier, artifacts and configuration; live-judge results are a separate experiment.

Regeneration Policy:
--------------------
This corpus represents the calibrated baseline decision gate. The frozen fixture
tests/fixtures/decision_replay_corpus.jsonl MUST NOT be regenerated to make a
failing test pass.

The corpus is ONLY regenerated alongside a deliberate, reviewed, and approved
governance decision change (e.g. recalibration, architectural upgrade, or model update).

To regenerate the corpus:
    python -m scripts.generate_decision_replay_corpus
    OR
    python tests/test_decision_replay.py --regenerate
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import pytest



CORPUS_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "decision_replay_corpus.jsonl")


def _load_corpus() -> List[Dict[str, Any]]:
    """Loads all test cases from the frozen regression corpus."""
    if not os.path.exists(CORPUS_FILE):
        pytest.fail(f"Decision replay corpus fixture missing at {CORPUS_FILE}")

    rows: List[Dict[str, Any]] = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Corpus syntax error in {CORPUS_FILE} line {line_num}: {exc}")
    return rows


def test_corpus_fixture_validity():
    """Validates that the frozen corpus fixture exists, is non-empty, and has valid SHA256 hashes."""
    rows = _load_corpus()
    assert len(rows) >= 200, f"Expected at least 200 corpus rows, got {len(rows)}"

    for row in rows:
        assert "id" in row, "Corpus row missing 'id'"
        assert "text" in row, f"Corpus row {row.get('id')} missing 'text'"
        assert "expected_risk" in row, f"Corpus row {row.get('id')} missing 'expected_risk'"
        assert "expected_source" in row, f"Corpus row {row.get('id')} missing 'expected_source'"
        assert "text_sha256" in row, f"Corpus row {row.get('id')} missing 'text_sha256'"

        computed_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        assert computed_sha == row["text_sha256"], (
            f"Corpus row {row['id']} text_sha256 mismatch: recorded {row['text_sha256']}, computed {computed_sha}"
        )


@pytest.mark.slow
def test_decision_replay_regression_gate(tmp_path):
    """Run in a fresh process so unit-test model state cannot change the oracle."""
    import subprocess
    from pathlib import Path

    output = tmp_path / "canonical-replay"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.canonical_replay", "--output", str(output)],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800,
    )
    if completed.returncode != 0:
        mismatch_path = output / "mismatches.json"
        mismatches = mismatch_path.read_text(encoding="utf-8") if mismatch_path.exists() else ""
        pytest.fail(
            "Canonical decision replay failed; degraded model tiers are not equivalent.\n"
            + completed.stdout + completed.stderr + mismatches
        )


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        from scripts.generate_decision_replay_corpus import generate_corpus
        generate_corpus()
    else:
        print("Run with pytest: pytest tests/test_decision_replay.py -v")
        print("To regenerate: python tests/test_decision_replay.py --regenerate")
