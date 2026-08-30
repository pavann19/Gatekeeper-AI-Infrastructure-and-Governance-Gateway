"""
Decision-Replay Regression Gate.

Re-runs a frozen stratified sample of ~300 rows from data/eval_suite.jsonl through
core.risk.assess_risk(text) and asserts that risk_level matches expected_risk exactly
with zero tolerance.

The corpus baseline was generated with NO judge backend reachable; 56 rows expect HIGH via judge_failure_fail_closed. On a host with a live judge, regenerate.

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

import core.risk as risk_mod
from core.risk import assess_risk


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
def test_decision_replay_regression_gate(monkeypatch):
    """
    Re-scores every row in the frozen corpus through assess_risk(text).
    Asserts risk_level matches expected_risk exactly with zero tolerance.
    On any mismatch, fails with a detailed diff.
    """
    rows = _load_corpus()
    assert len(rows) > 0, "Corpus is empty"

    # Bypass cache so the test validates the real decision engine directly
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)

    def _mock_fast_fail_post(*args, **kwargs):
        raise ConnectionError("Judge backend unreachable")

    # Mock judge HTTP calls and llama_guard path to fail-fast without connection timeout delays
    monkeypatch.setattr("core.semantic_judge.requests.post", _mock_fast_fail_post)
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration", lambda *a, **k: None)

    mismatches = []

    for row in rows:
        prompt_id = row["id"]
        text = row["text"]
        expected_risk = row["expected_risk"]
        expected_source = row.get("expected_source", "unknown")

        got_risk, details = assess_risk(text)
        got_source = details.get("source", "unknown")

        if got_risk != expected_risk:
            snippet = text[:80].replace("\n", " ")
            mismatches.append(
                f"- ID: {prompt_id}\n"
                f"  Snippet: {snippet!r}\n"
                f"  Expected Risk: {expected_risk} (Source: {expected_source})\n"
                f"  Got Risk:      {got_risk} (Source: {got_source})\n"
            )

    if mismatches:
        diff_report = "\n".join(mismatches)
        pytest.fail(
            f"Decision replay regression gate failed on {len(mismatches)}/{len(rows)} cases:\n\n"
            f"{diff_report}\n"
            "Corpus must NOT be regenerated without deliberate, reviewed decision changes."
        )


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        from scripts.generate_decision_replay_corpus import generate_corpus
        generate_corpus()
    else:
        print("Run with pytest: pytest tests/test_decision_replay.py -v")
        print("To regenerate: python tests/test_decision_replay.py --regenerate")
