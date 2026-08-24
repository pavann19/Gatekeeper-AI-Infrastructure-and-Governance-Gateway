"""
MCP compatibility adapter (Phase 6, Tool/Agent Gateway — the final
roadmap item, explicitly listed as "deferred to after the above is
solid"). Now that registry/schemas, allow/deny, risk-based approval,
sandboxed demo tools, and audit events all exist and are tested, this is
what makes them speak the Model Context Protocol's shape.

SCOPE: PROTOCOL-SHAPE COMPATIBILITY, NOT A TRANSPORT SERVER
----------------------------------------------------------------
This module translates between `core/tools.py`'s registry/`ToolSpec` and
MCP's own `Tool`/`CallToolResult` JSON shapes, and nothing else. It does
NOT implement an MCP server (no stdio transport, no JSON-RPC framing, no
SSE) — that is a real, separate piece of infrastructure with its own
integration surface (a process boundary, a wire protocol, a client
handshake) and belongs to whichever deployment actually needs to expose
Gatekeeper-governed tools over MCP, not to a compatibility layer built
speculatively ahead of one. This is the same "don't build the next thing
before something needs it" discipline every other item in this phase
already followed — `core/tools.py`'s own module docstring made the exact
same call about allow/deny and audit events before a real tool existed.

WHY THIS WAS ALREADY CHEAP
------------------------------
`core/tools.py::ToolSpec.parameters` was JSON-Schema-shaped from the very
first commit in this phase specifically so this step would be an
adapter, not a rewrite (see `ToolSpec`'s own docstring). MCP's `Tool.
inputSchema` is that same JSON Schema shape — `tool_spec_to_mcp` is
close to a field rename, not a translation layer with real logic in it.

MCP SHAPES REPRODUCED HERE (subset actually used)
------------------------------------------------------
    Tool:            {"name": str, "description": str, "inputSchema": <JSON Schema>}
    CallToolResult:  {"content": [{"type": "text", "text": str}], "isError": bool}

Gatekeeper's own decision vocabulary (BLOCK/REVIEW/ALLOW) has no MCP
equivalent — MCP's `CallToolResult` only distinguishes success from
error, nothing in between. A BLOCK or REVIEW is therefore reported as
`isError: true` with the reason as the content text; an MCP client has
no protocol-level way to distinguish "denied by policy" from "the tool
itself failed" without reading that text, which is a real, stated
limitation of MCP's shape, not something this adapter can paper over.
"""
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


def list_mcp_tools(registry: ToolRegistry = None) -> List[Dict[str, Any]]:
    """The MCP `tools/list` response body: every registered tool's MCP
    descriptor, in registration order."""
    reg = registry if registry is not None else get_tool_registry()
    return [tool_spec_to_mcp(spec) for spec in reg.list_tools()]


def handle_mcp_tool_call(capability: str, name: str, arguments: Dict[str, Any],
                         registry: ToolRegistry = None,
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
