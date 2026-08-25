"""
MCP HTTP and SSE transport server (Phase 6, Tool/Agent Gateway extension).
Provides a networked, HTTP/SSE transport for MCP clients with per-request
API key authentication, dynamic capability resolution, and tenant isolation.

Endpoints:
  - GET /mcp/sse: Opens an SSE stream and advertises the POST messages endpoint.
  - POST /mcp/messages?sessionId=<id>: Dispatches JSON-RPC 2.0 requests over HTTP.
  - POST /mcp/jsonrpc: Direct stateless JSON-RPC 2.0 endpoint for HTTP MCP clients.
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

    @app.get("/health")
    def health():
        return {"status": "ok", "transport": "mcp_http_sse"}

    @app.get("/mcp/sse")
    async def mcp_sse(request: Request):
        """
        SSE transport handshake per MCP specification:
        Sends an initial `endpoint` event containing the relative URI to post messages to.
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
                        data = await asyncio.wait_for(queue.get(), timeout=0.1)
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
        """Handles incoming JSON-RPC requests for an active SSE session."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            )

        response = await _dispatch_rpc(request, body)
        if response is None:
            # Notifications receive an empty 202 Accepted response per spec
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
