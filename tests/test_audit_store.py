"""Embedded-SQLite audit store."""
from __future__ import annotations

import json
import sqlite3

import pytest

from core import audit_store


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "audit.db")


def _input_entry(**over):
    e = {"timestamp": "2026-09-03T10:00:00", "event_type": "input_assessment",
         "request_id": "req-1", "tenant": "acme", "capability": "cap-x",
         "risk": "LOW", "decision": "ALLOW", "prompt_hash": "a" * 64,
         "semantic_score": 0.12, "source": "cache", "educational_context": False,
         "domain_score": None, "symbolic_triggered": False, "judge_invoked": False}
    e.update(over)
    return e


def test_write_and_read_input_assessment(db):
    audit_store.write("input_assessment", _input_entry(), path=db)
    out = audit_store.recent(path=db)
    assert out["scan_truncated"] is False
    assert len(out["events"]) == 1
    rec = out["events"][0]
    assert rec["event_type"] == "input_assessment"
    assert rec["request_id"] == "req-1"
    assert rec["decision"] == "ALLOW"
    assert rec["educational_context"] is False  # int 0 -> bool on read
    assert rec["semantic_score"] == pytest.approx(0.12)


def test_all_four_shapes_stay_separate(db):
    audit_store.write("input_assessment", _input_entry(request_id="r"), path=db)
    audit_store.write("output_assessment", {
        "timestamp": "2026-09-03T10:00:01", "request_id": "r", "tenant": "acme",
        "capability": "c", "decision": "BLOCK", "response_hash": "b" * 64,
        "pii_leakage": True, "secrets_detected": False, "toxicity_detected": False,
        "hallucination_detected": False, "system_prompt_leak_detected": False,
        "source": "output_guard"}, path=db)
    audit_store.write("gateway_call", {
        "timestamp": "2026-09-03T10:00:02", "request_id": "r", "tenant": "acme",
        "capability": "c", "provider": "openai", "model": "gpt-x", "success": True,
        "latency_ms": 42.0, "decision": "ALLOW", "usage": {"total_tokens": 10},
        "error": None}, path=db)
    audit_store.write("tool_call", {
        "timestamp": "2026-09-03T10:00:03", "request_id": "r", "tenant": "acme",
        "capability": "c", "tool": "search", "decision": "ALLOW",
        "risk_level": "LOW", "reason": "ok", "arguments_hash": "c" * 64,
        "success": True, "error": None}, path=db)

    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"input_assessment", "output_assessment", "gateway_call",
            "tool_call", "legacy_event"} <= tables

    trace = audit_store.by_request_id("r", path=db)["events"]
    assert [e["event_type"] for e in trace] == [
        "input_assessment", "output_assessment", "gateway_call", "tool_call"]
    # gateway usage round-trips as a dict, not a JSON string
    gw = trace[2]
    assert gw["usage"] == {"total_tokens": 10}
    assert trace[1]["pii_leakage"] is True


def test_indexes_exist_on_the_four_named_columns(db):
    audit_store.write("input_assessment", _input_entry(), path=db)
    conn = sqlite3.connect(db)
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    for col in ("request_id", "tenant", "timestamp", "event_type"):
        assert f"ix_input_assessment_{col}" in idx


def test_no_raw_content_columns(db):
    audit_store.write("input_assessment", _input_entry(), path=db)
    audit_store.write("output_assessment", {
        "timestamp": "t", "request_id": "r", "tenant": "x", "capability": "c",
        "decision": "ALLOW", "response_hash": "d" * 64, "pii_leakage": False,
        "secrets_detected": False, "toxicity_detected": False,
        "hallucination_detected": False, "system_prompt_leak_detected": False,
        "source": "s"}, path=db)
    conn = sqlite3.connect(db)
    banned = {"prompt", "response", "response_text", "text", "arguments",
              "argument", "content", "raw_prompt"}
    for table in ("input_assessment", "output_assessment", "gateway_call", "tool_call"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert not (cols & banned), f"{table} has a raw-content column: {cols & banned}"


def test_unknown_event_type_goes_to_legacy(db):
    audit_store.write("weird_old_shape", {
        "timestamp": "2020-01-01T00:00:00", "request_id": "old-1",
        "tenant": "unset", "some_dead_field": 7}, path=db)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT raw_json FROM legacy_event WHERE request_id='old-1'").fetchone()
    assert row is not None
    assert json.loads(row[0])["some_dead_field"] == 7
    rec = audit_store.by_request_id("old-1", path=db)["events"][0]
    assert rec["event_type"] == "legacy"
    assert rec["some_dead_field"] == 7


def test_recent_filters_by_tenant_and_event_type(db):
    audit_store.write("input_assessment", _input_entry(request_id="a", tenant="t1"), path=db)
    audit_store.write("input_assessment", _input_entry(request_id="b", tenant="t2"), path=db)
    only_t1 = audit_store.recent(tenant="t1", path=db)["events"]
    assert [e["request_id"] for e in only_t1] == ["a"]
    only_gw = audit_store.recent(event_types=["gateway_call"], path=db)["events"]
    assert only_gw == []


def test_migration_is_idempotent(tmp_path):
    jsonl = tmp_path / "audit.jsonl"
    db = tmp_path / "audit.db"
    lines = [
        {"timestamp": "2026-09-03T09:00:00", "event_type": "input_assessment",
         "request_id": "m1", "tenant": "acme", "capability": "c", "risk": "LOW",
         "decision": "ALLOW", "prompt_hash": "e" * 64, "semantic_score": 0.1,
         "source": "x", "educational_context": False, "domain_score": None,
         "symbolic_triggered": False, "judge_invoked": False},
        {"timestamp": "2024-01-01T00:00:00", "request_id": "legacy-1",
         "decision": "ALLOW"},  # no event_type -> legacy
    ]
    jsonl.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    from scripts.migrate_audit_to_sqlite import migrate
    s1 = migrate(str(jsonl), str(db))
    assert s1["imported"] == 2
    s2 = migrate(str(jsonl), str(db))
    assert s2["imported"] == 0
    assert s2["skipped_dup"] == 2

    assert len(audit_store.recent(path=str(db))["events"]) == 2
    legacy = audit_store.by_request_id("legacy-1", path=str(db))["events"]
    assert legacy[0]["event_type"] == "legacy"


def test_mirror_failure_does_not_raise_out_of_log_event(monkeypatch, tmp_path):
    """core/logger.py must swallow a SQLite mirror failure — the JSONL write
    is authoritative and the request must not fail on the mirror."""
    from core import logger as lg

    monkeypatch.setattr(lg.settings, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(lg.settings, "AUDIT_SQLITE_ENABLED", True)

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(audit_store, "write", boom)
    # must not raise
    lg.log_event("cap", "hello", "LOW", "ALLOW", {"request_id": "r9"})


def test_sqlite_disabled_is_a_noop(monkeypatch, tmp_path, db):
    from core import logger as lg
    monkeypatch.setattr(lg.settings, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(lg.settings, "AUDIT_SQLITE_ENABLED", False)
    monkeypatch.setattr(lg.settings, "AUDIT_DB_PATH", db)
    lg.log_event("cap", "hello", "LOW", "ALLOW", {"request_id": "r0"})
    import os
    assert not os.path.exists(db)  # nothing written when disabled
