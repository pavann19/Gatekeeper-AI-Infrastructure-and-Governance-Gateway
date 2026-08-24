"""
Tests for the tool-call audit event (Phase 6, "Audit events" roadmap
item): core.logger.log_tool_event as its own function, and its wiring
into core.tools.execute_tool for every decision branch.

Mirrors tests/test_output_audit_logging.py's own reasoning: a security
decision with no audit trail is a real gap, and the interesting property
to prove is that BLOCK/REVIEW are audited too, not just successful calls.
"""
import hashlib
import json
import logging
from unittest.mock import patch

import pytest

from core.logger import log_tool_event
from core.tools import ToolRegistry, execute_tool


class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def audit_records():
    """The audit logger has propagate=False (core/logger.py), so caplog
    (attached to the root logger) never sees its records. Attach a real
    handler directly to the actual "gatekeeper.audit" logger instead --
    the same singleton object log_tool_event's own `logging.getLogger(
    "gatekeeper.audit")` call returns, since Python's logging registry
    hands back the same Logger instance for a given name every time."""
    handler = _CapturingHandler()
    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        audit_logger.removeHandler(handler)


def make_spec(**overrides):
    from core.tools import ToolSpec
    defaults = dict(
        name="database.read",
        description="Read rows from a table.",
        parameters={
            "type": "object",
            "properties": {"table": {"type": "string"}},
            "required": ["table"],
        },
        risk_level="LOW",
        capability_required="GENERAL",
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


# --- log_tool_event: the function itself -------------------------------------

def test_emits_the_tool_call_event_type(audit_records):
    log_tool_event("GENERAL", "demo.echo", "ALLOW", risk_level="LOW",
                   reason="ok", arguments={"text": "hi"}, success=True)
    record = audit_records[-1]
    assert record.event_type == "tool_call"
    assert record.tool == "demo.echo"
    assert record.decision == "ALLOW"


def test_arguments_are_hashed_not_logged_verbatim(audit_records):
    log_tool_event("GENERAL", "demo.echo", "ALLOW", risk_level="LOW",
                   reason="ok", arguments={"text": "a secret value"}, success=True)
    record = audit_records[-1]
    assert "a secret value" not in json.dumps(vars(record), default=str)
    expected_hash = hashlib.sha256(
        json.dumps({"text": "a secret value"}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert record.arguments_hash == expected_hash


def test_hash_is_stable_regardless_of_key_order(audit_records):
    log_tool_event("GENERAL", "t", "ALLOW", "LOW", "ok", arguments={"a": 1, "b": 2})
    hash_1 = audit_records[-1].arguments_hash
    log_tool_event("GENERAL", "t", "ALLOW", "LOW", "ok", arguments={"b": 2, "a": 1})
    hash_2 = audit_records[-1].arguments_hash
    assert hash_1 == hash_2


def test_no_arguments_means_no_hash(audit_records):
    log_tool_event("GENERAL", "t", "BLOCK", None, "unknown tool")
    record = audit_records[-1]
    assert record.arguments_hash is None


def test_block_and_review_carry_no_success_or_error(audit_records):
    """success/error are only meaningful for ALLOW -- a BLOCK/REVIEW never
    reached a handler at all, so both must stay None, not be coerced to
    False."""
    log_tool_event("GENERAL", "t", "BLOCK", "HIGH", "access denied")
    record = audit_records[-1]
    assert record.success is None
    assert record.error is None


def test_handler_failure_reports_error_with_allow_decision(audit_records):
    log_tool_event("GENERAL", "t", "ALLOW", "LOW", "ok",
                   success=False, error="RuntimeError: boom")
    record = audit_records[-1]
    assert record.decision == "ALLOW"
    assert record.success is False
    assert record.error == "RuntimeError: boom"


# --- wiring: execute_tool emits exactly one event per call, every branch ----

@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(make_spec(), handler=lambda table: [f"row from {table}"])
    return reg


def test_unknown_tool_is_audited(registry):
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "does_not_exist", {}, registry=registry)
    mock_log.assert_called_once()
    assert mock_log.call_args.args[2] == "BLOCK"


def test_access_denial_is_audited(registry):
    reg = ToolRegistry()
    reg.register(make_spec(capability_required="INTERNAL"), handler=lambda table: [])
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {"table": "orders"}, registry=reg)
    mock_log.assert_called_once()
    assert mock_log.call_args.args[2] == "BLOCK"


def test_invalid_arguments_are_audited(registry):
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {}, registry=registry)  # missing "table"
    mock_log.assert_called_once()
    assert mock_log.call_args.args[2] == "BLOCK"


def test_review_decision_is_audited_and_handler_never_ran():
    calls = []
    reg = ToolRegistry()
    reg.register(make_spec(risk_level="HIGH"), handler=lambda table: calls.append(table))
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("INTERNAL", "database.read", {"table": "orders"}, registry=reg)
    mock_log.assert_called_once()
    assert mock_log.call_args.args[2] == "REVIEW"
    assert calls == []


def test_successful_call_is_audited_with_success_true(registry):
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {"table": "orders"}, registry=registry)
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["success"] is True


def test_handler_exception_is_audited_with_success_false(registry):
    reg = ToolRegistry()

    def boom(table):
        raise RuntimeError("downstream unreachable")

    reg.register(make_spec(), handler=boom)
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {"table": "orders"}, registry=reg)
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["success"] is False
    assert "RuntimeError" in mock_log.call_args.kwargs["error"]


def test_no_registered_handler_is_audited_as_block(registry):
    reg = ToolRegistry()
    reg.register(make_spec())  # no handler
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {"table": "orders"}, registry=reg)
    mock_log.assert_called_once()
    assert mock_log.call_args.args[2] == "BLOCK"


def test_exactly_one_event_per_call_never_zero_never_two(registry):
    """Every branch in execute_tool must log exactly once -- not
    forgotten, and not double-logged by an earlier return path plus a
    later one."""
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {"table": "orders"}, registry=registry)
    assert mock_log.call_count == 1


def test_tenant_and_request_id_flow_through(registry):
    with patch("core.logger.log_tool_event") as mock_log:
        execute_tool("GENERAL", "database.read", {"table": "orders"}, registry=registry,
                    tenant="acme", request_id="req-123")
    assert mock_log.call_args.kwargs["tenant"] == "acme"
    assert mock_log.call_args.kwargs["request_id"] == "req-123"
