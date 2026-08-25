"""
Tests for core/logger.py -- the WRITE side of the audit trail
(tests/test_activity.py exhaustively covers the READ side).

core/logger.py wires up its "gatekeeper.audit" FileHandler exactly once,
at import time, pointed at whatever core.config.settings.AUDIT_LOG_PATH
resolved to at that moment (guarded by `if not logger.handlers`).
Monkeypatching settings.AUDIT_LOG_PATH after import does NOT redirect
already-constructed handlers, so these tests swap the "gatekeeper.audit"
logger's handlers directly for the duration of each test (see the
`audit_file` fixture) -- this is the only way to observe real writes to
a tmp_path file without reaching into private state.
"""
import hashlib
import json
import logging

import pytest

import core.logger as logger_mod
from core.logger import (
    get_logger,
    log_event,
    log_gateway_event,
    log_output_event,
    log_tool_event,
)


AUDIT_LOGGER_NAME = "gatekeeper.audit"


@pytest.fixture
def audit_file(tmp_path):
    """Redirects the real "gatekeeper.audit" logger to a tmp_path file for
    the duration of the test, then restores whatever handlers were there
    before. This exercises the actual logging.FileHandler + JsonFormatter
    pipeline log_event/log_output_event/etc. write through -- not a mock."""
    path = tmp_path / "audit.jsonl"
    audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
    original_handlers = list(audit_logger.handlers)
    original_propagate = audit_logger.propagate

    for h in original_handlers:
        audit_logger.removeHandler(h)

    handler = logging.FileHandler(str(path), encoding="utf-8")
    from pythonjsonlogger import jsonlogger
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    )
    audit_logger.addHandler(handler)
    audit_logger.propagate = False
    audit_logger.setLevel(logging.INFO)

    try:
        yield path
    finally:
        audit_logger.removeHandler(handler)
        handler.close()
        for h in original_handlers:
            audit_logger.addHandler(h)
        audit_logger.propagate = original_propagate


def _read_lines(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- log_event (input_assessment) -------------------------------------------

def test_log_event_writes_valid_json_line_with_required_fields(audit_file):
    log_event(
        capability="GENERAL",
        prompt="how do I bake bread",
        risk="LOW",
        decision="ALLOW",
        metadata={"tenant": "acme", "request_id": "req-1", "source": "chat"},
    )
    records = _read_lines(audit_file)
    assert len(records) == 1
    r = records[0]
    assert r["event_type"] == "input_assessment"
    assert r["tenant"] == "acme"
    assert r["capability"] == "GENERAL"
    assert r["decision"] == "ALLOW"
    assert r["request_id"] == "req-1"
    assert r["risk"] == "LOW"
    assert "timestamp" in r and r["timestamp"]


def test_log_event_hashes_prompt_never_stores_raw_text(audit_file):
    """Compliance-critical: the audit trail must never contain the raw
    prompt text, only its hash -- a leaked/exported audit log must not
    itself become a data-exfiltration vector."""
    secret_prompt = "my SSN is 123-45-6789 and my password is hunter2"
    log_event(
        capability="GENERAL",
        prompt=secret_prompt,
        risk="LOW",
        decision="ALLOW",
    )
    records = _read_lines(audit_file)
    r = records[0]
    raw_line = json.dumps(r)
    assert secret_prompt not in raw_line
    assert "123-45-6789" not in raw_line
    assert "hunter2" not in raw_line
    expected_hash = hashlib.sha256(secret_prompt.encode("utf-8")).hexdigest()
    assert r["prompt_hash"] == expected_hash


def test_log_event_defaults_when_metadata_omitted(audit_file):
    log_event(capability="GENERAL", prompt="hi", risk="LOW", decision="ALLOW")
    r = _read_lines(audit_file)[0]
    assert r["tenant"] == "unset"
    assert r["request_id"] == "unset"
    assert r["source"] == "unknown"
    assert r["educational_context"] is False
    assert r["symbolic_triggered"] is False
    assert r["judge_invoked"] is False


def test_log_event_appends_not_overwrites(audit_file):
    log_event(capability="A", prompt="one", risk="LOW", decision="ALLOW",
               metadata={"request_id": "r1"})
    log_event(capability="B", prompt="two", risk="HIGH", decision="BLOCK",
               metadata={"request_id": "r2"})
    log_event(capability="C", prompt="three", risk="LOW", decision="ALLOW",
               metadata={"request_id": "r3"})
    records = _read_lines(audit_file)
    assert len(records) == 3
    assert [r["request_id"] for r in records] == ["r1", "r2", "r3"]
    assert [r["capability"] for r in records] == ["A", "B", "C"]


# --- log_output_event (output_assessment) -----------------------------------

def test_log_output_event_writes_valid_record(audit_file):
    log_output_event(
        capability="GENERAL",
        response_text="here is your answer",
        decision="BLOCK",
        metadata={"pii_leakage": True, "secrets_detected": False},
        tenant="acme",
        request_id="req-9",
    )
    r = _read_lines(audit_file)[0]
    assert r["event_type"] == "output_assessment"
    assert r["tenant"] == "acme"
    assert r["request_id"] == "req-9"
    assert r["capability"] == "GENERAL"
    assert r["decision"] == "BLOCK"
    assert r["pii_leakage"] is True
    assert r["secrets_detected"] is False


def test_log_output_event_hashes_response_never_stores_raw_text(audit_file):
    secret_response = "here is the confidential internal roadmap: launch date is X"
    log_output_event(
        capability="GENERAL",
        response_text=secret_response,
        decision="BLOCK",
        metadata={"pii_leakage": True},
    )
    r = _read_lines(audit_file)[0]
    raw_line = json.dumps(r)
    assert secret_response not in raw_line
    assert "confidential internal roadmap" not in raw_line
    expected_hash = hashlib.sha256(secret_response.encode("utf-8")).hexdigest()
    assert r["response_hash"] == expected_hash


def test_log_output_event_defaults(audit_file):
    log_output_event(capability="GENERAL", response_text="ok", decision="ALLOW")
    r = _read_lines(audit_file)[0]
    assert r["tenant"] == "unset"
    assert r["request_id"] == "unset"
    assert r["pii_leakage"] is False
    assert r["secrets_detected"] is False
    assert r["toxicity_detected"] is False
    assert r["hallucination_detected"] is False
    assert r["system_prompt_leak_detected"] is False


# --- log_gateway_event (gateway_call) ---------------------------------------

def test_log_gateway_event_writes_valid_record(audit_file):
    log_gateway_event(
        capability="GENERAL",
        provider="openai",
        model="gpt-4o",
        success=True,
        latency_ms=123.4,
        decision="ALLOW",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
        tenant="acme",
        request_id="req-5",
    )
    r = _read_lines(audit_file)[0]
    assert r["event_type"] == "gateway_call"
    assert r["tenant"] == "acme"
    assert r["request_id"] == "req-5"
    assert r["provider"] == "openai"
    assert r["model"] == "gpt-4o"
    assert r["success"] is True
    assert r["latency_ms"] == 123.4
    assert r["decision"] == "ALLOW"
    assert r["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}


def test_log_gateway_event_never_logs_prompt_or_response_content(audit_file):
    """The gateway event's own docstring says it deliberately carries no
    prompt/response content -- verify the record has no field that could
    smuggle raw text in (only counts/metadata fields are present)."""
    log_gateway_event(
        capability="GENERAL", provider="anthropic", model="claude",
        success=False, latency_ms=50.0, decision="BLOCK",
        error="rate_limited",
    )
    r = _read_lines(audit_file)[0]
    allowed_keys = {
        "timestamp", "event_type", "request_id", "tenant", "capability",
        "provider", "model", "success", "latency_ms", "decision", "usage",
        "error", "level", "name", "message",
    }
    assert set(r.keys()) <= allowed_keys
    assert r["error"] == "rate_limited"
    assert r["success"] is False


# --- log_tool_event (tool_call) ---------------------------------------------

def test_log_tool_event_writes_valid_record(audit_file):
    log_tool_event(
        capability="AGENT",
        tool_name="read_file",
        decision="ALLOW",
        risk_level="LOW",
        reason="authorized",
        arguments={"path": "/tmp/x.txt"},
        success=True,
        tenant="acme",
        request_id="req-7",
    )
    r = _read_lines(audit_file)[0]
    assert r["event_type"] == "tool_call"
    assert r["tenant"] == "acme"
    assert r["request_id"] == "req-7"
    assert r["capability"] == "AGENT"
    assert r["tool"] == "read_file"
    assert r["decision"] == "ALLOW"
    assert r["risk_level"] == "LOW"
    assert r["reason"] == "authorized"
    assert r["success"] is True


def test_log_tool_event_hashes_arguments_never_stores_raw_arguments(audit_file):
    """Per the module's own docstring: arguments are NEVER logged verbatim,
    only their SHA-256 hash -- a tool argument can carry the same sensitive
    content a prompt can (customer IDs, file paths, query strings)."""
    args = {"customer_id": "cust-42", "query": "SELECT * FROM secrets"}
    log_tool_event(
        capability="AGENT", tool_name="run_sql", decision="ALLOW",
        risk_level="MEDIUM", reason="ok", arguments=args, success=True,
    )
    r = _read_lines(audit_file)[0]
    raw_line = json.dumps(r)
    assert "cust-42" not in raw_line
    assert "SELECT * FROM secrets" not in raw_line
    assert "customer_id" not in raw_line
    expected_hash = hashlib.sha256(
        json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert r["arguments_hash"] == expected_hash


def test_log_tool_event_arguments_hash_stable_regardless_of_key_order(audit_file):
    log_tool_event(
        capability="AGENT", tool_name="t", decision="ALLOW", risk_level="LOW",
        reason="r", arguments={"a": 1, "b": 2},
    )
    log_tool_event(
        capability="AGENT", tool_name="t", decision="ALLOW", risk_level="LOW",
        reason="r", arguments={"b": 2, "a": 1},
    )
    records = _read_lines(audit_file)
    assert records[0]["arguments_hash"] == records[1]["arguments_hash"]


def test_log_tool_event_no_arguments_gives_none_hash(audit_file):
    log_tool_event(
        capability="AGENT", tool_name="t", decision="BLOCK", risk_level="HIGH",
        reason="unauthorized",
    )
    r = _read_lines(audit_file)[0]
    assert r["arguments_hash"] is None
    assert r["success"] is None
    assert r["error"] is None


def test_log_tool_event_block_decision_still_audited(audit_file):
    """A BLOCKed tool call must be as auditable as a successful one -- this
    is explicitly called out as more security-interesting in the module's
    docstring, not something that should silently vanish."""
    log_tool_event(
        capability="AGENT", tool_name="delete_all", decision="BLOCK",
        risk_level="CRITICAL", reason="not authorized for this capability",
    )
    r = _read_lines(audit_file)[0]
    assert r["decision"] == "BLOCK"
    assert r["risk_level"] == "CRITICAL"
    assert r["reason"] == "not authorized for this capability"


# --- Mixed event types append into one stream, as core.activity expects ----

def test_mixed_event_types_all_append_to_same_file(audit_file):
    log_event(capability="A", prompt="p", risk="LOW", decision="ALLOW",
               metadata={"request_id": "same-id"})
    log_output_event(capability="A", response_text="r", decision="ALLOW",
                      request_id="same-id")
    log_gateway_event(capability="A", provider="p", model="m", success=True,
                       latency_ms=1.0, decision="ALLOW", request_id="same-id")
    log_tool_event(capability="A", tool_name="t", decision="ALLOW",
                   risk_level="LOW", reason="ok", request_id="same-id")

    records = _read_lines(audit_file)
    assert len(records) == 4
    assert [r["event_type"] for r in records] == [
        "input_assessment", "output_assessment", "gateway_call", "tool_call",
    ]
    assert all(r["request_id"] == "same-id" for r in records)


# --- round-trip through core.activity's read side ---------------------------

def test_round_trips_through_get_recent_activity(audit_file):
    """Proves write/read compatibility: records written by core.logger's
    functions must be readable by core.activity.get_recent_activity, since
    that's the only consumer of this file in production."""
    import core.activity as activity_mod

    log_event(capability="GENERAL", prompt="p1", risk="LOW", decision="ALLOW",
               metadata={"tenant": "acme", "request_id": "r1"})
    log_tool_event(capability="AGENT", tool_name="t", decision="BLOCK",
                   risk_level="HIGH", reason="no", tenant="acme",
                   request_id="r2")

    result = activity_mod.get_recent_activity(tenant="acme", path=str(audit_file))
    ids = {e["request_id"] for e in result["events"]}
    assert ids == {"r1", "r2"}


# --- get_logger factory ------------------------------------------------------

def test_get_logger_returns_named_logger():
    lg = get_logger("gatekeeper.policy_loader")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "gatekeeper.policy_loader"


def test_get_logger_default_name_is_gatekeeper():
    lg = get_logger()
    assert lg.name == "gatekeeper"
    assert lg is logger_mod.logger


def test_get_logger_actually_logs_messages_at_expected_level(caplog):
    lg = get_logger("gatekeeper.domain_classifier")
    lg.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="gatekeeper.domain_classifier"):
        lg.info("classifier initialized")
        lg.warning("suspicious domain score")
    messages = [rec.message for rec in caplog.records]
    assert "classifier initialized" in messages
    assert "suspicious domain score" in messages
    levels = {rec.message: rec.levelname for rec in caplog.records}
    assert levels["classifier initialized"] == "INFO"
    assert levels["suspicious domain score"] == "WARNING"


def test_get_logger_used_by_multiple_modules_are_distinct_but_related():
    """core/policy_loader.py, core/domain_classifier.py,
    core/threat_centroid.py, core/privacy.py all call get_logger(name) for
    structured app logging -- verify distinct names produce distinct
    logger objects, all separate from the audit-only "gatekeeper.audit"
    logger (which must never receive ad-hoc app log messages)."""
    a = get_logger("gatekeeper.policy_loader")
    b = get_logger("gatekeeper.privacy")
    audit = logging.getLogger("gatekeeper.audit")
    assert a is not b
    assert a.name != audit.name
    assert b.name != audit.name


# --- audit log directory auto-creation --------------------------------------

def test_audit_log_directory_is_auto_created_when_missing(tmp_path, monkeypatch):
    """core/logger.py's module-level setup does
    `os.makedirs(os.path.dirname(AUDIT_LOG_PATH) or ".", exist_ok=True)`
    before opening the FileHandler -- i.e. it auto-creates missing parent
    directories rather than failing. Verified here by re-running that
    exact setup logic (not by re-importing the module, since the real
    module wires its handlers once at import time and must not be
    disturbed for other tests) against a nested path that does not exist
    yet, then confirming a FileHandler can open successfully."""
    nested_path = tmp_path / "nested" / "does" / "not" / "exist" / "audit.jsonl"
    assert not nested_path.parent.exists()

    import os as os_mod
    os_mod.makedirs(os_mod.path.dirname(str(nested_path)) or ".", exist_ok=True)
    assert nested_path.parent.exists()

    handler = logging.FileHandler(str(nested_path), encoding="utf-8")
    handler.close()
    assert nested_path.exists()


def test_module_source_uses_makedirs_with_exist_ok(tmp_path):
    """Guards the auto-create-not-fail contract at the source level: if a
    future edit swaps this for a bare os.mkdir (which raises on missing
    intermediate dirs) or drops exist_ok, this test catches the regression
    even without re-importing the module."""
    import inspect
    source = inspect.getsource(logger_mod)
    assert "os.makedirs" in source
    assert "exist_ok=True" in source
