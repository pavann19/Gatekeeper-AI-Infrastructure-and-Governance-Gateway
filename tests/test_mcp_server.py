"""
Tests for core/mcp_server.py -- the hand-rolled MCP stdio transport (see
its own module docstring for why hand-rolled rather than the official
SDK, which broke this project's pinned FastAPI dependency tree on
contact).

Two layers tested separately:
  - handle_request: pure dispatch logic, no I/O, one message in/out.
  - run_stdio_server: the real transport loop, driven with io.StringIO
    standing in for stdin/stdout -- genuinely exercises line-by-line
    JSON-RPC framing, not just the dispatch function in isolation.
"""
import io
import json

import pytest

from core.demo_tools import register_demo_tools
from core.mcp_server import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    handle_request,
    run_stdio_server,
)
from core.tools import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()
    register_demo_tools(reg)
    return reg


# --- handle_request: protocol dispatch ---------------------------------------

def test_initialize_returns_protocol_and_capabilities(registry):
    response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                              "GENERAL", registry=registry)
    assert response["id"] == 1
    assert "protocolVersion" in response["result"]
    assert response["result"]["capabilities"] == {"tools": {}}
    assert response["result"]["serverInfo"]["name"] == "gatekeeper"


def test_initialized_notification_gets_no_response(registry):
    response = handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"},
                              "GENERAL", registry=registry)
    assert response is None


def test_tools_list_returns_all_registered_tools(registry):
    response = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                              "GENERAL", registry=registry)
    names = {t["name"] for t in response["result"]["tools"]}
    assert names == {
        "demo.echo", "demo.calculator.add",
        "demo.database.query", "demo.database.delete",
    }


def test_tools_call_allowed_returns_success_content(registry):
    response = handle_request(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "demo.echo", "arguments": {"text": "hello"}}},
        "GENERAL", registry=registry,
    )
    assert response["id"] == 3
    assert response["result"]["isError"] is False
    assert response["result"]["content"] == [{"type": "text", "text": "hello"}]


def test_tools_call_denied_returns_is_error_true_not_a_jsonrpc_error(registry):
    """A Gatekeeper access denial is a normal MCP tool result
    (isError: true), not a JSON-RPC protocol-level error -- the request
    itself was well-formed, the TOOL CALL was denied."""
    response = handle_request(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "demo.database.query", "arguments": {"table": "orders"}}},
        "GENERAL", registry=registry,
    )
    assert "error" not in response
    assert response["result"]["isError"] is True


def test_tools_call_missing_name_is_invalid_params(registry):
    response = handle_request(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"arguments": {}}},
        "GENERAL", registry=registry,
    )
    assert response["error"]["code"] == INVALID_PARAMS


def test_tools_call_non_dict_arguments_is_invalid_params(registry):
    response = handle_request(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "demo.echo", "arguments": "not an object"}},
        "GENERAL", registry=registry,
    )
    assert response["error"]["code"] == INVALID_PARAMS


def test_unknown_method_is_method_not_found(registry):
    response = handle_request({"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
                              "GENERAL", registry=registry)
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_unknown_method_as_notification_gets_no_response(registry):
    """No 'id' means it's a notification -- JSON-RPC forbids replying to
    one even with an error."""
    response = handle_request({"jsonrpc": "2.0", "method": "resources/list"},
                              "GENERAL", registry=registry)
    assert response is None


def test_missing_method_field_is_invalid_request(registry):
    response = handle_request({"jsonrpc": "2.0", "id": 8}, "GENERAL", registry=registry)
    assert response["error"]["code"] == INVALID_REQUEST


def test_non_string_method_as_notification_is_dropped_silently(registry):
    response = handle_request({"jsonrpc": "2.0", "method": 123}, "GENERAL", registry=registry)
    assert response is None


def test_tenant_flows_through_to_the_audit_event(registry):
    from unittest.mock import patch
    with patch("core.logger.log_tool_event") as mock_log:
        handle_request(
            {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
             "params": {"name": "demo.echo", "arguments": {"text": "hi"}}},
            "GENERAL", registry=registry, tenant="acme-mcp",
        )
    assert mock_log.call_args.kwargs["tenant"] == "acme-mcp"


def test_capability_is_enforced_not_just_passed_through(registry):
    """A GENERAL-capability server process must not be able to reach an
    ELEVATED-only tool just because the MCP client asked -- there is no
    per-call auth in MCP's stdio transport, so the SERVER's fixed
    capability is the only enforcement point, and it must actually work."""
    response = handle_request(
        {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
         "params": {"name": "demo.database.query", "arguments": {"table": "orders"}}},
        "GENERAL", registry=registry,
    )
    assert response["result"]["isError"] is True
    response = handle_request(
        {"jsonrpc": "2.0", "id": 11, "method": "tools/call",
         "params": {"name": "demo.database.query", "arguments": {"table": "orders"}}},
        "ELEVATED", registry=registry,
    )
    assert response["result"]["isError"] is False


# --- run_stdio_server: the real transport loop -------------------------------

def _run(lines, capability="GENERAL", registry=None, tenant="mcp"):
    input_stream = io.StringIO("\n".join(lines) + "\n")
    output_stream = io.StringIO()
    run_stdio_server(capability=capability, registry=registry, tenant=tenant,
                     input_stream=input_stream, output_stream=output_stream)
    output_stream.seek(0)
    return [json.loads(line) for line in output_stream if line.strip()]


def test_stdio_loop_processes_a_full_session(registry):
    responses = _run([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "demo.echo", "arguments": {"text": "hi"}}}),
    ], registry=registry)
    # The notification produced no line -- exactly 3 responses for 4 inputs.
    assert len(responses) == 3
    assert responses[0]["id"] == 1
    assert responses[1]["id"] == 2
    assert responses[2]["result"]["content"] == [{"type": "text", "text": "hi"}]


def test_stdio_loop_survives_malformed_json_and_keeps_serving(registry):
    responses = _run([
        "{ this is not valid json",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    ], registry=registry)
    assert len(responses) == 2
    assert responses[0]["error"]["code"] == PARSE_ERROR
    assert responses[0]["id"] is None
    assert "result" in responses[1]  # the NEXT line still got served correctly


def test_stdio_loop_skips_blank_lines(registry):
    responses = _run([
        "",
        "   ",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    ], registry=registry)
    assert len(responses) == 1


def test_stdio_loop_rejects_non_object_json(registry):
    responses = _run(['["not", "an", "object"]'], registry=registry)
    assert responses[0]["error"]["code"] == INVALID_REQUEST


def test_stdio_loop_survives_a_handler_bug_and_keeps_serving(registry, monkeypatch):
    """A genuine internal error in dispatch (not a tool's own failure,
    which handle_mcp_tool_call already reports as isError -- an actual
    bug in this server's own code) must not take the whole session down."""
    import core.mcp_server as mcp_server_mod

    call_count = {"n": 0}
    real_handle_request = mcp_server_mod.handle_request

    def flaky(msg, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return real_handle_request(msg, *a, **kw)

    monkeypatch.setattr(mcp_server_mod, "handle_request", flaky)
    responses = _run([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ], registry=registry)
    assert responses[0]["error"]["code"] == INTERNAL_ERROR
    assert "result" in responses[1]  # the loop kept going after the crash


# --- Phase 8 hardening: bounded line length ------------------------------------
#
# `for line in stream` (the original implementation) buffers an entire
# line into memory before yielding it -- a single line with no newline
# would be an unbounded read. run_stdio_server now uses
# readline(MAX_LINE_BYTES + 1) instead, which bounds a single read
# regardless of whether a newline ever appears.

def test_oversized_line_is_rejected_without_buffering_it_whole(registry, monkeypatch):
    import core.mcp_server as mcp_server_mod
    monkeypatch.setattr(mcp_server_mod, "MAX_LINE_BYTES", 100)
    oversized = "x" * 500  # no newline within the first 100 bytes
    responses = _run([oversized], registry=registry)
    assert len(responses) == 1
    assert responses[0]["error"]["code"] == PARSE_ERROR
    assert responses[0]["id"] is None
    assert "exceeds" in responses[0]["error"]["message"]


def test_stdio_loop_resyncs_after_an_oversized_line_and_keeps_serving(registry, monkeypatch):
    """After dropping an oversized line, the NEXT real message must be
    parsed as its own message, not as a fragment glued onto the dropped
    one."""
    import core.mcp_server as mcp_server_mod
    monkeypatch.setattr(mcp_server_mod, "MAX_LINE_BYTES", 100)
    oversized = "x" * 500
    good = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    responses = _run([oversized, good], registry=registry)
    assert len(responses) == 2
    assert responses[0]["error"]["code"] == PARSE_ERROR
    assert "result" in responses[1]
    assert responses[1]["id"] == 1


def test_line_exactly_at_the_limit_is_still_accepted(registry, monkeypatch):
    import core.mcp_server as mcp_server_mod
    msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    monkeypatch.setattr(mcp_server_mod, "MAX_LINE_BYTES", len(msg) + 10)
    responses = _run([msg], registry=registry)
    assert len(responses) == 1
    assert "result" in responses[0]
