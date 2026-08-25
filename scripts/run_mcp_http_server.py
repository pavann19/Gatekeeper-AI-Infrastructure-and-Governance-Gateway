"""
CLI runner for the Gatekeeper MCP HTTP/SSE transport server.

Usage:
    python -m scripts.run_mcp_http_server --port 8001 --host 127.0.0.1
"""
import argparse
import uvicorn

from core.demo_tools import register_demo_tools
from core.mcp_http_server import create_mcp_app
from core.tools import get_tool_registry


def main():
    parser = argparse.ArgumentParser(description="Run Gatekeeper MCP HTTP/SSE Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument(
        "--demo-tools",
        action="store_true",
        help="Register core/demo_tools.py's sandboxed demo tools before serving (default: False).",
    )

    args = parser.parse_args()

    registry = get_tool_registry()
    if args.demo_tools:
        register_demo_tools(registry)

    app = create_mcp_app(registry=registry)
    print(f"Starting Gatekeeper MCP HTTP/SSE server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
