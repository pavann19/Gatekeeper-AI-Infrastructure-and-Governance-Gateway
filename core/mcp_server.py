"""
MCP stdio transport server (Phase 6, Tool/Agent Gateway's "real MCP
transport server" follow-up to `core/mcp_compat.py`'s protocol-shape
adapter).

WHY HAND-ROLLED, NOT THE OFFICIAL `mcp` SDK
------------------------------------------------
Tried the SDK first. Installing it pulled in a `starlette` major-version
bump (0.38.6 -> 1.6.0) incompatible with this project's pinned FastAPI
(`starlette<0.39.0`), cascading into `pydantic`/`uvicorn`/`h11` conflicts
that broke every API test module on contact. Reverted immediately,
confirmed the suite back to green before writing a line of this file.
MCP's stdio transport is JSON-RPC 2.0 over newline-delimited JSON on
stdin/stdout — a small, well-specified protocol subset — and `core/
mcp_compat.py` already has every piece of actual logic (list tools, call
a tool). What was missing is only message framing and method dispatch,
which is genuinely less code, and zero new dependencies, than working
around the SDK's conflicting dependency tree.

SCOPE: THE STDIO TRANSPORT, TOOLS ONLY
-------------------------------------------
Implements `initialize`, `notifications/initialized`, `tools/list`, and
`tools/call` — the subset MCP calls the "tools" capability, which is all
`core/mcp_compat.py` ever adapted. Resources, prompts, sampling, and any
other MCP capability are out of scope; nothing here claims to be a
general-purpose MCP server, only a tools-capable one.

CALLER IDENTITY: A FIXED CAPABILITY PER SERVER PROCESS
------------------------------------------------------------
MCP has no notion of Gatekeeper's own API-key-based capability
resolution — an MCP client (typically a trusted local process, e.g. a
desktop AI assistant) connects over stdio with no request-level
authentication at all. This server therefore runs at ONE fixed
capability for its entire process lifetime, set at startup
(`run_stdio_server(capability=...)`), the same trust-boundary shape
every real-world local MCP server already has: the process itself is
the security boundary, not each individual call within it. This is a
stated, deliberate scope decision, not an oversight — a deployment
wanting per-caller capability differentiation over MCP would need a
different transport (MCP's HTTP/SSE transport, with its own auth story),
which is explicitly not built here (see `core/mcp_compat.py`'s own
docstring on why a transport server was deferred until this exact point).
"""
from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Dict, Optional

from core.logger import get_logger
from core.mcp_compat import handle_mcp_tool_call, list_mcp_tools
from core.tools import ToolRegistry

logger = get_logger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "gatekeeper"
SERVER_VERSION = "1.0.0"

# Phase 8 hardening: `for line in stream` buffers an entire line into
# memory before yielding it, so one line with no newline is an unbounded
# read. This module's stated trust model (a stdio MCP client is a trusted
# local process, the process IS the security boundary) is why this was
# not treated as a network-facing vulnerability -- but a bounded read
# costs nothing and turns "a misbehaving client wedges this process by
# accident" into a clean, loud protocol error instead of unbounded memory
# growth, so it is worth having regardless of trust level. 1MB
# comfortably covers any real JSON-RPC message this server's own methods
# produce or accept (tools/call arguments are themselves capped at 100KB
# serialized -- see api/schemas.py::ToolCallRequest).
MAX_LINE_BYTES = 1_000_000

# Standard JSON-RPC 2.0 error codes (https://www.jsonrpc.org/specification).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _error_response(msg_id, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _result_response(msg_id, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def handle_request(msg: Dict[str, Any], capability: str, registry: ToolRegistry = None,
                   tenant: str = "mcp") -> Optional[Dict[str, Any]]:
    """
    Dispatches one already-parsed JSON-RPC message and returns the
    response dict, or None for a NOTIFICATION (no "id" field) — per the
    JSON-RPC 2.0 spec, a server MUST NOT reply to notifications, success
    or failure alike, so a malformed notification is logged and silently
    dropped rather than returned as an error with nowhere to send it.

    Pure and synchronous: no stdio I/O happens here, which is what makes
    this function directly testable without a subprocess or real pipes —
    `run_stdio_server` is the thin loop around it that actually reads and
    writes.
    """
    has_id = "id" in msg
    msg_id = msg.get("id")
    method = msg.get("method")

    if not isinstance(method, str):
        if not has_id:
            logger.warning(f"Dropping malformed notification (no valid method): {msg}")
            return None
        return _error_response(msg_id, INVALID_REQUEST, "Request must have a string 'method'.")

    params = msg.get("params") or {}

    if method == "initialize":
        return _result_response(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method == "notifications/initialized":
        # The client's ack that it received our `initialize` result. A
        # notification either way -- nothing to reply with, nothing to do.
        return None

    if method == "tools/list":
        return _result_response(msg_id, {"tools": list_mcp_tools(registry)})

    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str) or not name:
            if not has_id:
                return None
            return _error_response(msg_id, INVALID_PARAMS, "'params.name' must be a non-empty string.")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            if not has_id:
                return None
            return _error_response(msg_id, INVALID_PARAMS, "'params.arguments' must be an object.")

        result = handle_mcp_tool_call(capability, name, arguments, registry=registry,
                                      tenant=tenant, request_id=uuid.uuid4().hex)
        if not has_id:
            return None
        return _result_response(msg_id, result)

    if not has_id:
        logger.warning(f"Dropping notification for unknown method {method!r}.")
        return None
    return _error_response(msg_id, METHOD_NOT_FOUND, f"Unknown method: {method!r}")


def run_stdio_server(capability: str = "GENERAL", registry: ToolRegistry = None,
                     tenant: str = "mcp", input_stream=None, output_stream=None) -> None:
    """
    The actual transport loop: one JSON-RPC message per line on
    `input_stream` (default `sys.stdin`), one JSON-RPC response per line
    written to `output_stream` (default `sys.stdout`), flushed
    immediately — a client reading line-by-line must never block waiting
    for a buffered write. Runs until the input stream is exhausted
    (the client closed its side, e.g. the parent process exited).

    A line that isn't valid JSON produces a PARSE_ERROR response with
    `id: null` per the JSON-RPC spec (there is no request id to echo back
    when the request itself couldn't be parsed) — logged and continued,
    never crashes the loop, matching this project's consistent
    fail-closed-not-fail-crashed posture for a malformed input rather
    than an internal fault. A line longer than `MAX_LINE_BYTES` gets the
    same treatment (see that constant's own docstring) — read via
    `readline(MAX_LINE_BYTES + 1)`, which bounds a single read regardless
    of whether a newline ever appears, rather than `for line in stream`,
    which does not.
    """
    input_stream = input_stream if input_stream is not None else sys.stdin
    output_stream = output_stream if output_stream is not None else sys.stdout

    logger.info(f"MCP stdio server starting (capability={capability!r}, tenant={tenant!r}).")

    while True:
        raw = input_stream.readline(MAX_LINE_BYTES + 1)
        if raw == "":
            break  # EOF -- the client closed its side.

        if len(raw) > MAX_LINE_BYTES and not raw.endswith("\n"):
            # Hit the cap without finding a newline: this "line" is
            # oversized. Report it, then keep discarding reads until the
            # real newline is found (or EOF) so the NEXT readline() call
            # starts at a genuine line boundary again, rather than
            # treating an arbitrary later fragment as if it were a fresh
            # message.
            logger.warning(f"Dropping oversized MCP line (> {MAX_LINE_BYTES} bytes).")
            response = _error_response(None, PARSE_ERROR, f"Line exceeds {MAX_LINE_BYTES} byte limit.")
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()
            while not raw.endswith("\n") and raw != "":
                raw = input_stream.readline(MAX_LINE_BYTES + 1)
            continue

        line = raw.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            response = _error_response(None, PARSE_ERROR, f"Invalid JSON: {e}")
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()
            continue

        if not isinstance(msg, dict):
            response = _error_response(None, INVALID_REQUEST, "Request must be a JSON object.")
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()
            continue

        try:
            response = handle_request(msg, capability, registry=registry, tenant=tenant)
        except Exception as e:
            # A bug in dispatch/handling must not take the whole server
            # down mid-session -- report it as an internal error for
            # THIS message and keep serving the next one.
            logger.exception(f"Unhandled error processing MCP message: {msg}")
            response = _error_response(msg.get("id"), INTERNAL_ERROR, f"{type(e).__name__}: {e}") \
                if "id" in msg else None

        if response is not None:
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()

    logger.info("MCP stdio server stopped (input stream closed).")
