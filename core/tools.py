"""
Tool registry + access control (Phase 6, Tool/Agent Gateway — roadmap
items "Tool registry + schemas" and "Allow/deny, argument validation").

SCOPE, DELIBERATELY NARROW
--------------------------
This covers the registry, schema declaration, structural argument
validation (required fields present, declared types match), and
capability-based allow/deny (`check_tool_access`). It does NOT cover
risk-based approval requirements, sandboxed execution, or audit events —
those are separate, explicitly listed roadmap items, and semantic
argument validation ("this table name must exist") is inherently
per-tool, deferred until a real tool needs it. Building the remaining
items here, before a single real tool is registered to exercise any of
this, would repeat the exact mistake this project's evidence discipline
has consistently avoided elsewhere: several new subsystems stacked
before the first is used, none of them driven by a measured need. Ship
this, let it be used, let the next item's real requirements surface
from that.

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

# core/policy.py configures each capability's risk->decision mapping
# independently per tenant (nothing enforces that INTERNAL's mapping is
# ever a superset of ELEVATED's), but every policy actually shipped in
# this project follows the same ordering in practice: INTERNAL is at
# least as permissive as ELEVATED at every risk level, which is at least
# as permissive as GENERAL. Tool access uses that same observed ordering
# as an explicit minimum-privilege rank, documented as an assumption
# rather than a structural guarantee — a tenant policy CAN be configured
# to violate it, the same way core/policy.py's own JSON config could
# already configure GENERAL more permissively than INTERNAL if someone
# deliberately wrote it that way.
CAPABILITY_RANK = {"GENERAL": 0, "ELEVATED": 1, "INTERNAL": 2}

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


def check_tool_access(capability: str, spec: ToolSpec) -> Tuple[bool, str]:
    """
    Allow/deny: does a caller at `capability` meet the tool's declared
    `capability_required`? Same (ok, detail) shape as `validate_arguments`
    and `Detector.available()` — a denial is an expected outcome to
    report cleanly, not an exception.

    An unrecognised `capability` (should not happen if it came from
    `core/auth.py::resolve_principal`, which only ever returns a
    validated tier, but this function must not assume its caller did
    that) is treated as UNRANKED and denied — fail closed, the same
    direction every other unknown-input path in this project fails,
    rather than defaulting to the lowest rank and silently reasoning
    about a capability that was never actually validated.
    """
    caller_rank = CAPABILITY_RANK.get(capability)
    if caller_rank is None:
        return False, f"unrecognised capability {capability!r}; access denied"

    required_rank = CAPABILITY_RANK[spec.capability_required]
    if caller_rank < required_rank:
        return False, (
            f"tool {spec.name!r} requires {spec.capability_required!r} capability, "
            f"caller has {capability!r}"
        )
    return True, "ok"


# Decision vocabulary reuses core/policy.py's own action names (minus
# RESTRICT, which has no obvious meaning for a tool CALL -- a prompt can
# be answered more cautiously; a database write either happens or it
# doesn't) rather than inventing a parallel one. Ordering mirrors the
# same BLOCK > REVIEW > ALLOW severity this project already uses for
# combining two verdicts (see core/policy.py's VALID_ACTIONS docstring
# and api/main.py's _SEVERITY dict).
VALID_TOOL_DECISIONS = ("BLOCK", "REVIEW", "ALLOW")


def decide_tool_call(capability: str, spec: ToolSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    The full pre-execution decision for one tool call: access control,
    then structural validation, then risk-based approval — in that
    order, cheapest and most decisive check first, same ordering
    principle `api/main.py`'s own request handlers already follow
    (rate limit before token quota before the expensive detection work).

    HIGH-risk tools always require REVIEW, even for a caller whose
    capability already clears `check_tool_access` — access control
    answers "may this caller use this tool at all", approval answers "is
    THIS SPECIFIC call safe enough to run without a human looking at it
    first", and conflating them would mean an INTERNAL caller's HIGH-risk
    call (e.g. a production delete) executes with no human in the loop
    just because they're allowed to invoke the tool in general. This
    mirrors Phase 4's own REVIEW semantics: "neither auto-allowed nor
    auto-blocked."

    Returns a dict, not a bare decision string, so a caller can log or
    display `reason` and `tool` alongside `decision` without a second
    lookup — the same reasoning `core/fusion.py`'s `fused_threat_score`
    returns a dict rather than a bare score.

    NOT wired into an execution endpoint or the review queue yet — there
    is no real tool call path to enforce this against, and forcing a
    tool-call review into `core/review_queue.py`'s prompt-hash-shaped
    `ReviewRecord` would be exactly the "one shape serving two different
    questions" mistake `core/logger.py`'s three distinct audit-event
    functions were built to avoid — that concern was about the AUDIT
    LOG's shape specifically (a compliance record answering a fixed set
    of questions), not the review QUEUE's, which already generalises: a
    `ReviewRecord`'s `prompt_hash` field is documented as "an identifying
    hash, never raw content" — a tool call's name+arguments hash fits
    that same contract exactly, even though the field name is inherited
    from Phase 4's prompt-oriented origin. `POST /api/v1/tools/call`
    (added once this function had a real caller to wire it to) does
    exactly that.

    `risk_level` is included in every returned dict — not just useful
    context, but what a caller enqueuing a REVIEW needs to populate
    `ReviewRecord.risk` without a second lookup against the registry.
    """
    access_ok, access_detail = check_tool_access(capability, spec)
    if not access_ok:
        return {"decision": "BLOCK", "reason": access_detail, "tool": spec.name,
                "risk_level": spec.risk_level}

    valid_ok, valid_detail = validate_arguments(spec, arguments)
    if not valid_ok:
        return {"decision": "BLOCK", "reason": valid_detail, "tool": spec.name,
                "risk_level": spec.risk_level}

    if spec.risk_level == "HIGH":
        return {
            "decision": "REVIEW",
            "reason": f"tool {spec.name!r} is HIGH risk; human approval required",
            "tool": spec.name,
            "risk_level": spec.risk_level,
        }

    return {"decision": "ALLOW", "reason": "ok", "tool": spec.name, "risk_level": spec.risk_level}


class ToolRegistry:
    """
    Maps tool name -> (ToolSpec, handler). Mirrors `core/detectors.py`'s
    registry shape deliberately: a plain dict, register-then-look-up
    usage, one place that owns "what tools exist" — consistency with
    this codebase's other pluggable-component registry beats a
    marginally different interface here.

    `handler` is OPTIONAL and separate from the spec on purpose: a spec
    is a declared contract (what a tool is, its risk, who may call it) —
    something a policy or a UI needs to reason about even for a tool this
    process cannot itself execute (e.g. one a different service runs).
    Registering a spec without a handler is a normal, supported state;
    `execute` on such a tool fails with a clear message rather than a
    silent no-op or an `AttributeError` from calling `None`.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, Any] = {}

    def register(self, spec: ToolSpec, handler=None) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        if handler is not None:
            self._handlers[spec.name] = handler
        logger.info(f"Registered tool {spec.name!r} (risk={spec.risk_level}, "
                   f"capability_required={spec.capability_required}, "
                   f"handler={'yes' if handler is not None else 'no'})")

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(self._tools)}")
        return self._tools[name]

    def get_handler(self, name: str):
        """Returns the registered handler, or None if this tool has no
        handler (spec-only, or executed elsewhere). Never raises for an
        unregistered handler — raises KeyError only if `name` itself
        isn't a registered tool, same as `get`."""
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(self._tools)}")
        return self._handlers.get(name)

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def execute_tool(capability: str, name: str, arguments: Dict[str, Any],
                 registry: "ToolRegistry" = None,
                 tenant: str = "unset", request_id: str = "unset") -> Dict[str, Any]:
    """
    The full call path: look up the tool, decide (access + validation +
    risk), and only invoke the handler if the decision is ALLOW. Every
    outcome — BLOCK and REVIEW included, not only ALLOW — is audited via
    `core.logger.log_tool_event` before returning; an unauthorized or
    malformed call is itself a security event worth recording, arguably
    more interesting than a routine successful one.

    A handler NEVER runs for BLOCK or REVIEW — this is the actual
    enforcement point Phase 6 exists to build, not just a decision
    function nobody consults. "Sandboxed" for the demo tools registered
    in `core/demo_tools.py` means what it should always mean for a
    handler with unknown provenance: it operates on in-memory, fake data
    only, with no filesystem, network, or real-database access — the
    sandbox is a property of what a handler is ALLOWED to touch, not
    something this function itself enforces at the process level (no
    subprocess isolation, no seccomp, no container). A handler that
    reached into real infrastructure would defeat the sandbox regardless
    of what this function does; the safety property lives in which
    handlers get registered, not in `execute_tool`'s own code.

    A handler that raises is reported as an execution error, distinct
    from a security decision — the tool call was ALLOWED to attempt
    running, and failed on its own terms (bad input the schema didn't
    catch, a downstream dependency down), not because Gatekeeper decided
    against it. Conflating the two would make a flaky tool look like a
    security block in the audit trail.

    `tenant`/`request_id` default to "unset" the same way every other
    audit-emitting function in this codebase does — "unset" lets a query
    distinguish "no request context existed" from "this caller resolved
    to a default", same reasoning `core/logger.py::log_event`'s own
    docstring gives for that field. `POST /api/v1/tools/call` supplies
    real values for both.
    """
    from core.logger import log_tool_event

    reg = registry if registry is not None else _registry
    try:
        spec = reg.get(name)
    except KeyError as e:
        detail = str(e)
        log_tool_event(capability, name, "BLOCK", risk_level=None, reason=detail,
                       arguments=arguments, tenant=tenant, request_id=request_id)
        return {"decision": "BLOCK", "reason": detail, "tool": name, "risk_level": None}

    result = decide_tool_call(capability, spec, arguments)
    if result["decision"] != "ALLOW":
        log_tool_event(capability, name, result["decision"], risk_level=spec.risk_level,
                       reason=result["reason"], arguments=arguments,
                       tenant=tenant, request_id=request_id)
        return result

    handler = reg.get_handler(name)
    if handler is None:
        detail = f"tool {name!r} has no registered handler"
        log_tool_event(capability, name, "BLOCK", risk_level=spec.risk_level, reason=detail,
                       arguments=arguments, tenant=tenant, request_id=request_id)
        return {"decision": "BLOCK", "reason": detail, "tool": name, "risk_level": spec.risk_level}

    try:
        output = handler(**arguments)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log_tool_event(capability, name, "ALLOW", risk_level=spec.risk_level,
                       reason=result["reason"], arguments=arguments, success=False,
                       error=error, tenant=tenant, request_id=request_id)
        return {**result, "error": error}

    log_tool_event(capability, name, "ALLOW", risk_level=spec.risk_level,
                   reason=result["reason"], arguments=arguments, success=True,
                   tenant=tenant, request_id=request_id)

    return {**result, "output": output}


# Module-level singleton, same reasoning as core/rate_limit.py's
# assess_rate_limiter and core/detectors.py's registry: one shared
# instance for the process, populated at startup by whatever deployment
# registers its own tools.
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _registry
