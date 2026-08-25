"""
Additional edge-case coverage for core/mcp_compat.py, complementing
tests/test_mcp_compat.py. Focus areas not already covered there:
schema-shape edge cases (empty parameters, nested objects, arrays,
Phase 8's maxLength constraint), multi-tool list ordering/content
fidelity, and malformed/unexpected MCP-side call input.
"""

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


# --- schema translation edge cases --------------------------------------------

def test_empty_parameters_schema_translates_verbatim():
    spec = make_spec(parameters={"type": "object", "properties": {}})
    mcp_tool = tool_spec_to_mcp(spec)
    assert mcp_tool["inputSchema"] == {"type": "object", "properties": {}}


def test_nested_object_schema_preserved_exactly():
    nested = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {
                    "range": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                        },
                        "required": ["min", "max"],
                    }
                },
                "required": ["range"],
            }
        },
        "required": ["filter"],
    }
    spec = make_spec(parameters=nested)
    mcp_tool = tool_spec_to_mcp(spec)
    assert mcp_tool["inputSchema"] == nested
    assert mcp_tool["inputSchema"]["properties"]["filter"]["properties"]["range"][
        "properties"]["min"]["type"] == "number"


def test_array_schema_with_item_type_preserved():
    schema = {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["ids"],
    }
    spec = make_spec(parameters=schema)
    mcp_tool = tool_spec_to_mcp(spec)
    assert mcp_tool["inputSchema"]["properties"]["ids"]["type"] == "array"
    assert mcp_tool["inputSchema"]["properties"]["ids"]["items"] == {"type": "integer"}


def test_max_length_constraint_survives_translation():
    """Phase 8 added JSON-Schema `maxLength` enforcement in
    core/tools.py::validate_arguments. The MCP-facing schema must carry
    that constraint through unchanged so an MCP client can see it."""
    schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "maxLength": 2048}},
        "required": ["url"],
    }
    spec = make_spec(name="http.get", parameters=schema)
    mcp_tool = tool_spec_to_mcp(spec)
    assert mcp_tool["inputSchema"]["properties"]["url"]["maxLength"] == 2048


def test_max_length_violation_still_enforced_through_mcp_call():
    """The adapter must not bypass validate_arguments's maxLength check --
    confirm a too-long string is rejected end-to-end via handle_mcp_tool_call,
    not just that the schema field is copied."""
    schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "maxLength": 5}},
        "required": ["url"],
    }
    reg = ToolRegistry()
    reg.register(make_spec(name="http.get", parameters=schema, risk_level="LOW"),
                 handler=lambda url: f"fetched {url}")
    result = handle_mcp_tool_call("GENERAL", "http.get", {"url": "way-too-long-a-url"}, registry=reg)
    assert result["isError"] is True
    assert "maxLength" in result["content"][0]["text"]


def test_max_length_within_bound_succeeds_through_mcp_call():
    schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "maxLength": 10}},
        "required": ["url"],
    }
    reg = ToolRegistry()
    reg.register(make_spec(name="http.get", parameters=schema, risk_level="LOW"),
                 handler=lambda url: f"fetched {url}")
    result = handle_mcp_tool_call("GENERAL", "http.get", {"url": "short"}, registry=reg)
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": "fetched short"}]


# --- list_mcp_tools: multi-tool fidelity --------------------------------------

def test_list_preserves_each_tools_full_schema_independently():
    reg = ToolRegistry()
    reg.register(make_spec(name="tool.a", parameters={
        "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}))
    reg.register(make_spec(name="tool.b", parameters={
        "type": "object", "properties": {"y": {"type": "integer"}}, "required": ["y"]}))
    mcp_tools = {t["name"]: t for t in list_mcp_tools(reg)}
    assert mcp_tools["tool.a"]["inputSchema"]["properties"] == {"x": {"type": "string"}}
    assert mcp_tools["tool.b"]["inputSchema"]["properties"] == {"y": {"type": "integer"}}


def test_list_registration_order_is_preserved():
    reg = ToolRegistry()
    reg.register(make_spec(name="third"))
    reg.register(make_spec(name="first"))
    reg.register(make_spec(name="second"))
    names = [t["name"] for t in list_mcp_tools(reg)]
    assert names == ["third", "first", "second"]


# --- malformed / unexpected MCP-side input ------------------------------------

def test_arguments_missing_required_field_is_error():
    reg = ToolRegistry()
    reg.register(make_spec(name="database.read", risk_level="LOW"),
                 handler=lambda table: f"rows from {table}")
    result = handle_mcp_tool_call("GENERAL", "database.read", {}, registry=reg)
    assert result["isError"] is True
    assert "content" in result and len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"


def test_arguments_with_unexpected_extra_field_does_not_crash():
    reg = ToolRegistry()
    reg.register(make_spec(name="database.read", risk_level="LOW"),
                 handler=lambda table: f"rows from {table}")
    result = handle_mcp_tool_call("GENERAL", "database.read",
                                  {"table": "orders", "unexpected_field": "x"}, registry=reg)
    # Extra fields beyond the schema aren't rejected by this project's
    # validator; the call proceeds and the handler only sees declared args
    # it actually accepts. Confirm no crash and a well-formed response.
    assert isinstance(result, dict)
    assert set(result.keys()) == {"content", "isError"}


def test_empty_name_is_treated_as_unknown_tool():
    reg = ToolRegistry()
    result = handle_mcp_tool_call("GENERAL", "", {}, registry=reg)
    assert result["isError"] is True
    assert isinstance(result["content"][0]["text"], str)


def test_call_result_always_has_exactly_content_and_is_error_keys():
    """Guard the CallToolResult shape itself for both success and error
    paths -- no stray keys leaking from execute_tool's richer dict."""
    reg = ToolRegistry()
    reg.register(make_spec(name="database.read", risk_level="LOW"),
                 handler=lambda table: "ok")
    ok = handle_mcp_tool_call("GENERAL", "database.read", {"table": "orders"}, registry=reg)
    err = handle_mcp_tool_call("GENERAL", "does.not.exist", {}, registry=reg)
    assert set(ok.keys()) == {"content", "isError"}
    assert set(err.keys()) == {"content", "isError"}


def test_content_text_is_always_a_string_even_for_non_string_output():
    reg = ToolRegistry()
    reg.register(make_spec(name="numeric.tool", risk_level="LOW"),
                 handler=lambda table: {"rows": 42})
    result = handle_mcp_tool_call("GENERAL", "numeric.tool", {"table": "orders"}, registry=reg)
    assert result["isError"] is False
    assert isinstance(result["content"][0]["text"], str)
    assert "42" in result["content"][0]["text"]
