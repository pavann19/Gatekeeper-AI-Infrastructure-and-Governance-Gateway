"""
MCP HTTP and SSE transport server (Phase 6, Tool/Agent Gateway extension).
Provides a networked, HTTP/SSE transport for MCP clients with per-request
API key authentication, dynamic capability resolution, and tenant isolation.

ARCHITECTURE & PROTOCOL CONFORMANCE
------------------------------------
Implements the Model Context Protocol (MCP) 2024-11-05 SSE transport specification:
1. SSE Handshake (`GET /mcp/sse`):
   - Opens a persistent text/event-stream connection.
   - Emits an initial `endpoint` event advertising `/mcp/messages?sessionId=<uuid>`.
   - Relays JSON-RPC responses and server events asynchronously over the SSE stream (`event: message`).
2. Session-Relayed Message Endpoint (`POST /mcp/messages?sessionId=<uuid>`):
   - Accepts client JSON-RPC requests.
   - Routes the execution through `core.mcp_server.handle_request`.
   - Enqueues the JSON-RPC response to the corresponding session's SSE stream and returns HTTP 202 Accepted.
   - Returns HTTP 404 if the session ID is missing from active connections or expired.
   - Falls back to returning the response directly in the HTTP body if `sessionId` is omitted.
3. Direct JSON-RPC Endpoint (`POST /mcp/jsonrpc`):
   - Stateless HTTP POST endpoint returning JSON-RPC responses directly (HTTP 200).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.auth import auth_required, resolve_principal
from core.logger import get_logger
from core.mcp_server import handle_request
from core.tenancy import resolve_tenant
from core.tools import ToolRegistry, get_tool_registry

logger = get_logger(__name__)


def create_mcp_app(registry: Optional[ToolRegistry] = None) -> FastAPI:
    """Creates a FastAPI application configured for MCP HTTP/SSE transport."""
    app = FastAPI(title="Gatekeeper MCP Server", version="1.0.0")
    tool_registry = registry or get_tool_registry()

    # Active SSE sessions: session_id -> asyncio.Queue for server-to-client events
    sessions: Dict[str, asyncio.Queue] = {}
    app.state.sessions = sessions

    @app.get("/health")
    def health():
        return {"status": "ok", "transport": "mcp_http_sse"}

    @app.get("/mcp/sse")
    async def mcp_sse(request: Request):
        """
        SSE transport handshake per MCP specification:
        Sends an initial `endpoint` event containing the relative URI to post messages to,
        then streams all JSON-RPC responses enqueued for this session.
        """
        session_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        sessions[session_id] = queue

        async def event_generator():
            try:
                # 1. Send the endpoint discovery event
                endpoint_url = f"/mcp/messages?sessionId={session_id}"
                yield f"event: endpoint\ndata: {endpoint_url}\n\n"

                # 2. Keep stream alive / yield any server push messages
                while True:
                    try:
                        if await request.is_disconnected():
                            break
                    except Exception:
                        break

                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=0.5)
                        yield f"event: message\ndata: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        if request.headers.get("X-Test-Stream-Once"):
                            break
                        yield ": ping\n\n"
            finally:
                sessions.pop(session_id, None)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-ID": session_id,
            },
        )

    async def _dispatch_rpc(request: Request, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Resolve authentication from Authorization header or X-API-Key
        auth_header = request.headers.get("Authorization")
        if not auth_header and request.headers.get("X-API-Key"):
            auth_header = f"Bearer {request.headers.get('X-API-Key')}"

        principal = resolve_principal(authorization=auth_header)
        if auth_required() and not principal.authenticated:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Present a valid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        tenant_config = resolve_tenant(principal.tenant)
        if tenant_config.suspended:
            raise HTTPException(
                status_code=403,
                detail=f"Tenant '{principal.tenant}' is suspended.",
            )

        # Dispatch via standard core.mcp_server logic
        response = handle_request(
            msg=body,
            capability=principal.capability,
            registry=tool_registry,
            tenant=principal.tenant,
        )
        return response

    @app.post("/mcp/messages")
    async def mcp_messages(request: Request, sessionId: Optional[str] = Query(None)):
        """
        Handles incoming JSON-RPC requests for an active SSE session.
        Per MCP SSE transport spec:
        - If sessionId is provided and active: relays response over the open SSE stream and returns HTTP 202.
        - If sessionId is provided but unknown/expired: returns HTTP 404.
        - If sessionId is omitted: returns the JSON-RPC response directly (HTTP 200).
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            )

        if sessionId is not None:
            queue = sessions.get(sessionId)
            if queue is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"SSE session {sessionId!r} not found or expired.",
                )

            response = await _dispatch_rpc(request, body)
            if response is not None:
                await queue.put(response)
            return JSONResponse(status_code=202, content={"status": "accepted"})

        # Direct stateless fallback if sessionId is omitted
        response = await _dispatch_rpc(request, body)
        if response is None:
            return JSONResponse(status_code=202, content={})
        return JSONResponse(status_code=200, content=response)

    @app.post("/mcp/jsonrpc")
    async def mcp_jsonrpc(request: Request):
        """Direct stateless JSON-RPC POST endpoint."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            )

        response = await _dispatch_rpc(request, body)
        if response is None:
            return JSONResponse(status_code=202, content={})
        return JSONResponse(status_code=200, content=response)

    return app
