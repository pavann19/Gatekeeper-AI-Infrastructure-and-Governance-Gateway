"""Map registered tool specifications and execution into MCP request/response shapes.

Transport-specific admission remains outside this compatibility layer; shared
tool capability, risk, review and audit controls are enforced on execution."""
from __future__ import annotations

from typing import Any, Dict, List

from core.tools import ToolRegistry, ToolSpec, execute_tool, get_tool_registry


def tool_spec_to_mcp(spec: ToolSpec) -> Dict[str, Any]:
    """
    One `ToolSpec` -> one MCP `Tool` descriptor. `risk_level` and
    `capability_required` are Gatekeeper-specific access-control
    metadata with no place in MCP's schema — deliberately dropped here,
    not because they stop mattering (`execute_tool` still enforces both
    on every call) but because MCP's `Tool` shape has no field for them
    and inventing a non-standard extension field would be exactly the
    kind of made-up compatibility this module exists to avoid.
    """
    return {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": spec.parameters,
    }


def list_mcp_tools(registry: ToolRegistry | None = None) -> List[Dict[str, Any]]:
    """The MCP `tools/list` response body: every registered tool's MCP
    descriptor, in registration order."""
    reg = registry if registry is not None else get_tool_registry()
    return [tool_spec_to_mcp(spec) for spec in reg.list_tools()]


def handle_mcp_tool_call(capability: str, name: str, arguments: Dict[str, Any],
                         registry: ToolRegistry | None = None,
                         tenant: str = "unset", request_id: str = "unset") -> Dict[str, Any]:
    """
    The MCP `tools/call` response body for one call — routes through
    `execute_tool` (access control, structural validation, risk-based
    approval, execution, audit) exactly as any other caller would, then
    reshapes the result into `CallToolResult`.

    BLOCK and REVIEW both become `isError: true`, since MCP's shape has
    no third state — see module docstring's "MCP shapes reproduced here"
    section for why that's a real protocol limitation, not an oversight
    here. The full Gatekeeper decision (BLOCK vs REVIEW vs a handler's
    own execution error) is still fully audited via `execute_tool`'s own
    `log_tool_event` call either way; only the MCP-facing response loses
    that distinction, because MCP's response shape has nowhere to put it.
    """
    result = execute_tool(capability, name, arguments, registry=registry,
                          tenant=tenant, request_id=request_id)

    if result["decision"] in ("BLOCK", "REVIEW"):
        return {
            "content": [{"type": "text", "text": f"{result['decision']}: {result['reason']}"}],
            "isError": True,
        }

    if "error" in result:
        return {
            "content": [{"type": "text", "text": f"Tool execution failed: {result['error']}"}],
            "isError": True,
        }

    return {
        "content": [{"type": "text", "text": str(result.get("output"))}],
        "isError": False,
    }
