"""
MCP HTTP and SSE transport server (Phase 6, Tool/Agent Gateway extension).
Provides a networked, HTTP/SSE transport for MCP clients with per-request
API key authentication, dynamic capability resolution, tenant isolation,
request size bounding, and distributed/local rate limiting.

ARCHITECTURE & PROTOCOL CONFORMANCE
------------------------------------
Implements the Model Context Protocol (MCP) 2024-11-05 SSE transport specification:
1. SSE Handshake (`GET /mcp/sse`):
   - Opens a persistent text/event-stream connection.
   - Emits an initial `endpoint` event advertising `/mcp/messages?sessionId=<uuid>`.
   - Relays JSON-RPC responses and server events asynchronously over the SSE stream (`event: message`).
2. Session-Relayed Message Endpoint (`POST /mcp/messages?sessionId=<uuid>`):
   - Accepts client JSON-RPC requests up to 100KB.
   - Validates session, checks per-tenant rate limits, and routes the execution through `core.mcp_server.handle_request`.
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
from core.config import settings
from core.logger import get_logger
from core.mcp_server import handle_request
from core.rate_limit import assess_rate_limiter, bucket_parameters
from core.tenancy import resolve_tenant
from core.tools import ToolRegistry, get_tool_registry

logger = get_logger(__name__)

# Max request payload size: 100KB (matches Phase 8 schema hardening)
MAX_PAYLOAD_BYTES = 100 * 1024


def _client_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"


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

    async def _read_and_validate_body(request: Request) -> Dict[str, Any]:
        """Reads raw body, enforces 100KB size cap, and parses JSON object."""
        raw_body = await request.body()
        if len(raw_body) > MAX_PAYLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Request payload exceeds maximum size of {MAX_PAYLOAD_BYTES} bytes.",
            )

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Parse error: malformed JSON payload.",
            )

        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid request: top-level JSON-RPC payload must be a JSON object.",
            )

        return body

    async def _dispatch_rpc(request: Request, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1. Resolve authentication from Authorization header or X-API-Key
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

        # 2. Check tenant suspension
        tenant_config = resolve_tenant(principal.tenant)
        if tenant_config.suspended:
            raise HTTPException(
                status_code=403,
                detail=f"Tenant '{principal.tenant}' is suspended.",
            )

        # 3. Check rate limits
        if settings.RATE_LIMIT_ENABLED:
            if principal.authenticated:
                identity = f"key:{principal.key_id}"
                rpm = settings.RATE_LIMIT_AUTHENTICATED_RPM
                if tenant_config.rate_limit_rpm is not None:
                    rpm = tenant_config.rate_limit_rpm
            else:
                identity = f"ip:{_client_address(request)}"
                rpm = settings.RATE_LIMIT_ANONYMOUS_RPM

            capacity, refill = bucket_parameters(rpm, settings.RATE_LIMIT_BURST_SECONDS)
            allowed, retry_after = assess_rate_limiter.check(identity, capacity, refill)
            if not allowed:
                headers = {"Retry-After": str(int(retry_after) + 1)}
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Limit is {rpm:g} requests per minute.",
                    headers=headers,
                )

        # 4. Dispatch via standard core.mcp_server logic
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
        body = await _read_and_validate_body(request)

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
        body = await _read_and_validate_body(request)
        response = await _dispatch_rpc(request, body)
        if response is None:
            return JSONResponse(status_code=202, content={})
        return JSONResponse(status_code=200, content=response)

    return app
