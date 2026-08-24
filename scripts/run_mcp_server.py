"""
CLI entrypoint for `core.mcp_server`'s stdio transport — the thing an
MCP client (e.g. a desktop AI assistant's config) actually launches as a
subprocess.

CALLER IDENTITY
----------------
`--capability` sets the fixed capability this server process runs at for
its entire lifetime — see `core/mcp_server.py`'s own docstring for why
that is a deliberate scope boundary (MCP's stdio transport has no
per-call authentication of its own), not a missing feature.

DEMO TOOLS
-----------
`--demo-tools` registers `core/demo_tools.py`'s four sandboxed tools so
this is runnable and testable standalone. A real deployment omits the
flag and registers its own tools into `core.tools.get_tool_registry()`
before importing this module (or forks this script) instead.

Usage:
    python -m scripts.run_mcp_server --demo-tools
    python -m scripts.run_mcp_server --capability ELEVATED --tenant acme
"""
import argparse

from core.mcp_server import run_stdio_server
from core.tools import VALID_CAPABILITIES


def main():
    parser = argparse.ArgumentParser(description="Run Gatekeeper's MCP stdio server")
    parser.add_argument("--capability", default="GENERAL", choices=VALID_CAPABILITIES,
                        help="Fixed capability this server process runs at (default: GENERAL).")
    parser.add_argument("--tenant", default="mcp",
                        help="Tenant label recorded on every tool-call audit event (default: mcp).")
    parser.add_argument("--demo-tools", action="store_true",
                        help="Register core/demo_tools.py's sandboxed demo tools before serving.")
    args = parser.parse_args()

    if args.demo_tools:
        from core.demo_tools import register_demo_tools
        register_demo_tools()

    run_stdio_server(capability=args.capability, tenant=args.tenant)


if __name__ == "__main__":
    main()
