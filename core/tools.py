"""
Tool registry (Phase 6, Tool/Agent Gateway — roadmap item "Tool registry
+ schemas", the first of six listed items).

SCOPE, DELIBERATELY NARROW
--------------------------
This is ONLY the registry, the schema declaration, and basic structural
argument validation (required fields present, declared types match).
Allow/deny by capability, risk-based approval requirements, sandboxed
execution, and audit events are separate, explicitly listed roadmap
items. Building them here — before a single tool exists to exercise this
piece — would repeat the exact mistake this project's evidence
discipline has consistently avoided elsewhere: several new subsystems
stacked before the first is solid, none of them driven by a measured
need. Ship this, let it be used, let the next item's real requirements
surface from that.

SCHEMA FORMAT: JSON-SCHEMA-SHAPED, ON PURPOSE
------------------------------------------------
`parameters` follows JSON Schema's object-with-properties shape (the same
convention OpenAI function-calling and MCP's own tool schemas use). The
roadmap explicitly defers MCP compatibility until after this item and
"Allow/deny, argument validation" are solid — starting from a
JSON-Schema-compatible shape now means that future step is a
compatibility adapter, not a rewrite of every registered tool's schema.

NOT A JSON SCHEMA VALIDATOR
------------------------------
`validate_arguments` checks required-field presence and declared
top-level types (string/number/integer/boolean/array/object, plus
`enum`) — not the full JSON Schema spec (nested schemas, `minimum`,
`pattern`, `oneOf`, etc.). Pulling in a JSON Schema library for a
type-and-required-fields check would be the same mistake `evaluation/
metrics.py` avoided by hand-rolling its own bootstrap CI rather than
adding scipy: correct depth for what has an actual caller today, not
speculative completeness for validation rules no registered tool uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from core.logger import get_logger

logger = get_logger(__name__)

# Mirrors this project's existing HIGH/MEDIUM/LOW vocabulary
# (core/risk.py, core/policy.py) rather than inventing a parallel one —
# a tool's risk level and a prompt's risk level should read the same way
# to anyone operating this system.
VALID_RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

# Mirrors core/config.py's CAPABILITY_* tiers for the same reason: "which
# capability can call this tool" is the same access-control vocabulary
# as "which capability gets which policy outcome" elsewhere in this
# project, not a second, tool-specific permission system.
VALID_CAPABILITIES = ("GENERAL", "ELEVATED", "INTERNAL")

_JSON_SCHEMA_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class ToolSpec:
    """
    One registered tool's identity, contract, and access requirements.

    `parameters` is a JSON-Schema object schema:
        {"type": "object",
         "properties": {"table": {"type": "string", "enum": [...]}, ...},
         "required": ["table"]}

    `risk_level` and `capability_required` are declared, not derived —
    the same reasoning `core/detectors.py`'s `Detector.targets` uses:
    what a tool does and who may call it are facts about the tool, known
    at registration time, not inferred from its schema.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    risk_level: str = "MEDIUM"
    capability_required: str = "GENERAL"

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("tool name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError(f"tool {self.name!r} must have a non-empty description")
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(
                f"tool {self.name!r}: risk_level must be one of {VALID_RISK_LEVELS}, "
                f"got {self.risk_level!r}"
            )
        if self.capability_required not in VALID_CAPABILITIES:
            raise ValueError(
                f"tool {self.name!r}: capability_required must be one of "
                f"{VALID_CAPABILITIES}, got {self.capability_required!r}"
            )
        if self.parameters.get("type") != "object":
            raise ValueError(
                f"tool {self.name!r}: parameters must be a JSON-Schema object "
                f"schema (\"type\": \"object\"), got {self.parameters.get('type')!r}"
            )
        if not isinstance(self.parameters.get("properties", {}), dict):
            raise ValueError(f"tool {self.name!r}: parameters['properties'] must be an object")


def validate_arguments(spec: ToolSpec, arguments: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Structural validation only — required fields present, declared types
    match. NOT semantic validation ("this table name must exist in the
    database"): that is either the tool's own concern at call time, or a
    future item once a real tool exists to need it. Returns (ok, detail),
    same shape as `Detector.available()` elsewhere in this codebase, for
    the same reason: a boolean plus a human-readable reason, not an
    exception, since a malformed tool call is an expected input to
    reject cleanly, not a programming error.
    """
    if not isinstance(arguments, dict):
        return False, "arguments must be a JSON object"

    props = spec.parameters.get("properties", {})
    required = spec.parameters.get("required", [])

    for field_name in required:
        if field_name not in arguments:
            return False, f"missing required argument: {field_name!r}"

    for key, value in arguments.items():
        if key not in props:
            return False, f"unknown argument: {key!r}"

        expected_type = props[key].get("type")
        # bool is a subclass of int in Python -- checked BEFORE the
        # isinstance test below, not nested inside its failure branch:
        # isinstance(True, int) is True, so a JSON `true` would otherwise
        # silently satisfy an "integer"/"number" field.
        if expected_type in ("integer", "number") and isinstance(value, bool):
            return False, f"argument {key!r} must be of type {expected_type!r}, got bool"

        py_type = _JSON_SCHEMA_TYPES.get(expected_type)
        if py_type is not None and not isinstance(value, py_type):
            return False, (
                f"argument {key!r} must be of type {expected_type!r}, "
                f"got {type(value).__name__}"
            )

        enum = props[key].get("enum")
        if enum is not None and value not in enum:
            return False, f"argument {key!r} must be one of {enum!r}, got {value!r}"

    return True, "ok"


class ToolRegistry:
    """
    Maps tool name -> ToolSpec. Mirrors `core/detectors.py`'s registry
    shape deliberately: a plain dict, register-then-look-up usage, one
    place that owns "what tools exist" — consistency with this
    codebase's other pluggable-component registry beats a marginally
    different interface here.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        logger.info(f"Registered tool {spec.name!r} (risk={spec.risk_level}, "
                   f"capability_required={spec.capability_required})")

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(self._tools)}")
        return self._tools[name]

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Module-level singleton, same reasoning as core/rate_limit.py's
# assess_rate_limiter and core/detectors.py's registry: one shared
# instance for the process, populated at startup by whatever deployment
# registers its own tools.
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _registry
