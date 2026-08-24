"""
Tests for core/mcp_compat.py — the final Phase 6 item, "MCP
compatibility". Scope matches the module's own docstring: protocol-SHAPE
translation between core/tools.py and MCP's Tool/CallToolResult JSON,
not a transport server. These tests verify the shapes are correct and
that execute_tool's full decision pipeline (access, validation, risk,
audit) still runs underneath every call routed through this adapter.
"""
import pytest

from core.demo_tools import register_demo_tools
from core.mcp_compat import handle_mcp_tool_call, list_mcp_tools, tool_spec_to_mcp
from core.tools import ToolRegistry, ToolSpec


def make_spec(**overrides):
    defaults = dict(
        name="database.read",
        description="Read rows from a table.",
        parameters={
            "type": "object",
            "properties": {"table": {"type": "string", "enum": ["orders", "customers"]}},
            "required": ["table"],
        },
        risk_level="MEDIUM",
        capability_required="GENERAL",
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    register_demo_tools(reg)
    return reg


# --- tool_spec_to_mcp: shape translation --------------------------------------

def test_produces_the_mcp_tool_shape():
    spec = make_spec()
    mcp_tool = tool_spec_to_mcp(spec)
    assert set(mcp_tool.keys()) == {"name", "description", "inputSchema"}
    assert mcp_tool["name"] == "database.read"
    assert mcp_tool["description"] == "Read rows from a table."


def test_input_schema_is_the_spec_parameters_verbatim():
    """core/tools.py::ToolSpec.parameters was JSON-Schema-shaped from the
    start specifically so this would be near-verbatim, not a real
    translation -- confirm that's actually true, not just asserted in a
    docstring."""
    spec = make_spec()
    mcp_tool = tool_spec_to_mcp(spec)
    assert mcp_tool["inputSchema"] == spec.parameters


def test_risk_and_capability_metadata_are_not_leaked_into_mcp_shape():
    """MCP's Tool schema has no field for these -- they must not appear
    as an invented extension, which no real MCP client would understand."""
    spec = make_spec(risk_level="HIGH", capability_required="INTERNAL")
    mcp_tool = tool_spec_to_mcp(spec)
    assert "risk_level" not in mcp_tool
    assert "capability_required" not in mcp_tool
    assert "riskLevel" not in mcp_tool  # camelCase variant, just in case


# --- list_mcp_tools -----------------------------------------------------------

def test_lists_every_registered_tool(registry):
    mcp_tools = list_mcp_tools(registry)
    names = {t["name"] for t in mcp_tools}
    assert names == {
        "demo.echo", "demo.calculator.add",
        "demo.database.query", "demo.database.delete",
    }


def test_empty_registry_lists_no_tools():
    assert list_mcp_tools(ToolRegistry()) == []


def test_defaults_to_the_shared_registry():
    from core.tools import get_tool_registry
    shared = get_tool_registry()
    spec = make_spec(name="mcp_shared_test_tool")
    shared.register(spec)
    try:
        names = {t["name"] for t in list_mcp_tools()}
        assert "mcp_shared_test_tool" in names
    finally:
        del shared._tools["mcp_shared_test_tool"]


# --- handle_mcp_tool_call: routes through execute_tool's full pipeline ------

def test_allowed_call_returns_success_with_output_as_text(registry):
    result = handle_mcp_tool_call("GENERAL", "demo.echo", {"text": "hello"}, registry=registry)
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": "hello"}]


def test_access_denial_becomes_is_error_true(registry):
    """demo.database.query requires ELEVATED -- a GENERAL caller must be
    denied, and MCP has no non-error way to express that."""
    result = handle_mcp_tool_call("GENERAL", "demo.database.query",
                                  {"table": "orders"}, registry=registry)
    assert result["isError"] is True
    assert "BLOCK" in result["content"][0]["text"]


def test_high_risk_review_becomes_is_error_true(registry):
    """REVIEW has no MCP equivalent either -- also isError: true, with
    the reason distinguishing it in the text from a straightforward
    denial. The underlying decision (REVIEW, not BLOCK) is still fully
    audited by execute_tool even though MCP's own response can't carry
    that distinction."""
    result = handle_mcp_tool_call("INTERNAL", "demo.database.delete",
                                  {"table": "orders", "row_id": 1}, registry=registry)
    assert result["isError"] is True
    assert "REVIEW" in result["content"][0]["text"]


def test_invalid_arguments_become_is_error_true(registry):
    result = handle_mcp_tool_call("GENERAL", "demo.echo", {}, registry=registry)
    assert result["isError"] is True


def test_unknown_tool_becomes_is_error_true(registry):
    result = handle_mcp_tool_call("INTERNAL", "does_not_exist", {}, registry=registry)
    assert result["isError"] is True
    assert "does_not_exist" in result["content"][0]["text"]


def test_handler_exception_becomes_is_error_true_distinct_from_denial():
    reg = ToolRegistry()

    def boom(table):
        raise RuntimeError("downstream unreachable")

    reg.register(make_spec(risk_level="LOW"), handler=boom)
    result = handle_mcp_tool_call("GENERAL", "database.read", {"table": "orders"}, registry=reg)
    assert result["isError"] is True
    assert "Tool execution failed" in result["content"][0]["text"]
    assert "RuntimeError" in result["content"][0]["text"]


def test_successful_call_is_still_audited(registry):
    """Routing through the MCP adapter must not bypass the audit event
    the direct execute_tool path already guarantees."""
    from unittest.mock import patch
    with patch("core.logger.log_tool_event") as mock_log:
        handle_mcp_tool_call("GENERAL", "demo.echo", {"text": "hi"}, registry=registry)
    mock_log.assert_called_once()


def test_tenant_and_request_id_flow_through(registry):
    from unittest.mock import patch
    with patch("core.logger.log_tool_event") as mock_log:
        handle_mcp_tool_call("GENERAL", "demo.echo", {"text": "hi"}, registry=registry,
                             tenant="acme", request_id="req-42")
    assert mock_log.call_args.kwargs["tenant"] == "acme"
    assert mock_log.call_args.kwargs["request_id"] == "req-42"
