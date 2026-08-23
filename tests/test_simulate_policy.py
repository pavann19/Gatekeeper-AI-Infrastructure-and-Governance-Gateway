"""
Tests for scripts/simulate_policy.py's core logic (Phase 3, Policy-as-Code:
"Simulation"). Following this codebase's existing convention of not
building CLI-argument-parsing tests for scripts/ tools (none of the
existing ones have them) — these test the pure functions a CLI wraps.
"""
import json

from core.policy import PolicyStore, policy_decision
from scripts.simulate_policy import load_input_assessment_records


def test_load_input_assessment_records_keeps_only_rows_with_risk(tmp_path):
    """Output-assessment audit records (core/logger.py::log_output_event)
    have no `risk` field and must be skipped -- this tool simulates INPUT
    policy only."""
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps({"event_type": "input_assessment", "capability": "GENERAL",
                    "risk": "HIGH", "decision": "BLOCK"}) + "\n"
        + json.dumps({"event_type": "output_assessment", "capability": "GENERAL",
                      "decision": "BLOCK"}) + "\n",
        encoding="utf-8",
    )
    records = load_input_assessment_records(str(path))
    assert len(records) == 1
    assert records[0]["risk"] == "HIGH"


def test_load_input_assessment_records_skips_malformed_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "not valid json\n"
        + json.dumps({"capability": "GENERAL", "risk": "LOW", "decision": "ALLOW"}) + "\n"
        + "\n",  # blank line
        encoding="utf-8",
    )
    records = load_input_assessment_records(str(path))
    assert len(records) == 1


def test_load_input_assessment_records_missing_file_exits(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        load_input_assessment_records(str(tmp_path / "does_not_exist.jsonl"))


def test_replay_detects_a_decision_that_would_change(tmp_path):
    """End-to-end of the actual comparison logic simulate_policy.py's main()
    performs, without going through argv/CLI plumbing."""
    old_policy_path = tmp_path / "old.json"
    new_policy_path = tmp_path / "new.json"
    old_policy_path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"MEDIUM": "RESTRICT"}}}},
    }), encoding="utf-8")
    new_policy_path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"MEDIUM": "BLOCK"}}}},  # stricter
    }), encoding="utf-8")

    candidate_store = PolicyStore(path=str(new_policy_path))
    candidate_store.load()

    historical_record = {"capability": "GENERAL", "risk": "MEDIUM",
                         "tenant": "default", "decision": "RESTRICT"}

    new_decision, _ = policy_decision(
        historical_record["capability"], historical_record["risk"],
        historical_record["tenant"], store=candidate_store,
    )

    assert new_decision == "BLOCK"
    assert new_decision != historical_record["decision"]


def test_replay_confirms_unchanged_decisions_stay_unchanged(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}},
    }), encoding="utf-8")

    candidate_store = PolicyStore(path=str(policy_path))
    candidate_store.load()

    historical_record = {"capability": "GENERAL", "risk": "HIGH",
                         "tenant": "default", "decision": "BLOCK"}

    new_decision, _ = policy_decision(
        historical_record["capability"], historical_record["risk"],
        historical_record["tenant"], store=candidate_store,
    )
    assert new_decision == historical_record["decision"]
