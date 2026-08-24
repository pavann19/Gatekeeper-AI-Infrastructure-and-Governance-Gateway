"""
Sandboxed demo tools (Phase 6, Tool/Agent Gateway roadmap item
"Sandboxed demo tools") — the first real tools registered against
`core/tools.py`'s registry, existing to exercise the whole pipeline
(access control, structural validation, risk-based approval, execution)
end to end against something concrete, not just unit-tested in isolation.

WHAT "SANDBOXED" MEANS HERE
-----------------------------
Every handler below touches ONLY an in-memory, fake dataset defined in
this file — no filesystem, no network, no real database, no subprocess.
This is not a runtime sandbox (no container, no seccomp, no process
isolation) enforced BY `core/tools.py::execute_tool`; that function's own
docstring is explicit about this: the safety property lives in which
handlers get registered, not in code that runs around them. These
handlers are safe because there is nothing behind them TO damage, which
is exactly what a set of demo tools should be — illustrative of the
pipeline, not a claim about sandboxing arbitrary untrusted code.

WHY THESE FOUR, AT THESE RISK LEVELS
----------------------------------------
Chosen to exercise every branch `decide_tool_call` actually has:

    demo.echo                LOW,    GENERAL   -- the trivial ALLOW path
    demo.calculator.add      LOW,    GENERAL   -- ALLOW with real arguments
    demo.database.query      MEDIUM, ELEVATED  -- capability gate matters
    demo.database.delete     HIGH,   INTERNAL  -- triggers REVIEW even for
                                                    an INTERNAL caller who
                                                    clears access control

NOT REGISTERED AUTOMATICALLY
--------------------------------
Importing this module does not touch the shared registry — a production
deployment should not get demo tools by default just because this module
happens to be importable. `register_demo_tools(registry=None)` is the
explicit opt-in, the same "opt-in, not automatic" shape
`core/tenancy.py`'s tenant configuration and `core/policy.py`'s YAML
support already use elsewhere in this project.
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.tools import ToolRegistry, ToolSpec, get_tool_registry

# The "database" every demo.database.* tool operates on. Module-level and
# mutable on purpose (delete needs somewhere to actually remove a row
# from) but entirely in-memory -- restarting the process resets it, and
# nothing outside this module can observe or affect it.
_FAKE_DATABASE: Dict[str, List[Dict[str, Any]]] = {
    "orders": [
        {"id": 1, "customer": "acme", "total": 42.50},
        {"id": 2, "customer": "globex", "total": 17.00},
    ],
    "customers": [
        {"id": 1, "name": "acme"},
        {"id": 2, "name": "globex"},
    ],
}


def _echo(text: str) -> str:
    return text


def _add(a: float, b: float) -> float:
    return a + b


def _database_query(table: str, limit: int = 10) -> List[Dict[str, Any]]:
    rows = _FAKE_DATABASE.get(table, [])
    return rows[:limit]


def _database_delete(table: str, row_id: int) -> Dict[str, Any]:
    rows = _FAKE_DATABASE.get(table, [])
    before = len(rows)
    _FAKE_DATABASE[table] = [r for r in rows if r.get("id") != row_id]
    deleted = before - len(_FAKE_DATABASE[table])
    return {"table": table, "id": row_id, "deleted": deleted}


DEMO_TOOL_SPECS = [
    ToolSpec(
        name="demo.echo",
        description="Returns the text it was given, unchanged. The trivial ALLOW path.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        risk_level="LOW",
        capability_required="GENERAL",
    ),
    ToolSpec(
        name="demo.calculator.add",
        description="Adds two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        risk_level="LOW",
        capability_required="GENERAL",
    ),
    ToolSpec(
        name="demo.database.query",
        description="Reads rows from the in-memory demo database.",
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["orders", "customers"]},
                "limit": {"type": "integer"},
            },
            "required": ["table"],
        },
        risk_level="MEDIUM",
        capability_required="ELEVATED",
    ),
    ToolSpec(
        name="demo.database.delete",
        description="Deletes a row from the in-memory demo database by id. "
                    "HIGH risk -- always requires human review, even for an "
                    "INTERNAL caller.",
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["orders", "customers"]},
                "row_id": {"type": "integer"},
            },
            "required": ["table", "row_id"],
        },
        risk_level="HIGH",
        capability_required="INTERNAL",
    ),
]

_DEMO_HANDLERS = {
    "demo.echo": _echo,
    "demo.calculator.add": _add,
    "demo.database.query": _database_query,
    "demo.database.delete": _database_delete,
}


def register_demo_tools(registry: ToolRegistry = None) -> ToolRegistry:
    """
    Registers all four demo tools. Explicit opt-in — see module
    docstring for why this isn't automatic on import. Returns the
    registry it registered into, so a caller can chain or pass a fresh
    `ToolRegistry()` in tests without needing to also import it
    separately.
    """
    reg = registry if registry is not None else get_tool_registry()
    for spec in DEMO_TOOL_SPECS:
        if spec.name not in reg:
            reg.register(spec, handler=_DEMO_HANDLERS[spec.name])
    return reg
