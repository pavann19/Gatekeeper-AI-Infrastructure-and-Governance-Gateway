"""
Tests for core/tools.py — Phase 6's first item, "Tool registry + schemas".

Scope matches the module's own docstring: registry correctness and
structural argument validation. No allow/deny, approval, sandboxing, or
audit wiring here — those are separate, not-yet-built roadmap items.
"""
import pytest

from core.tools import (
    ToolRegistry,
    ToolSpec,
    check_tool_access,
    decide_tool_call,
    get_tool_registry,
    validate_arguments,
)


def make_spec(**overrides):
    defaults = dict(
        name="database.read",
        description="Read rows from a table.",
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["orders", "customers"]},
                "limit": {"type": "integer"},
            },
            "required": ["table"],
        },
        risk_level="MEDIUM",
        capability_required="GENERAL",
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


# --- ToolSpec validation ------------------------------------------------------

def test_valid_spec_constructs_cleanly():
    spec = make_spec()
    assert spec.name == "database.read"
    assert spec.risk_level == "MEDIUM"


def test_empty_name_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        make_spec(name="")


def test_whitespace_only_name_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        make_spec(name="   ")


def test_empty_description_rejected():
    with pytest.raises(ValueError, match="description"):
        make_spec(description="")


def test_invalid_risk_level_rejected():
    with pytest.raises(ValueError, match="risk_level"):
        make_spec(risk_level="CRITICAL")


def test_invalid_capability_rejected():
    with pytest.raises(ValueError, match="capability_required"):
        make_spec(capability_required="SUPERADMIN")


def test_non_object_parameters_rejected():
    with pytest.raises(ValueError, match="type.*object"):
        make_spec(parameters={"type": "array"})


def test_missing_properties_key_defaults_to_empty_object():
    """A tool with no arguments at all is valid — `properties` is optional
    in JSON Schema when there's nothing to declare."""
    spec = make_spec(parameters={"type": "object"})
    ok, detail = validate_arguments(spec, {})
    assert ok is True


def test_non_dict_properties_rejected():
    with pytest.raises(ValueError, match="properties"):
        make_spec(parameters={"type": "object", "properties": "not a dict"})


def test_default_parameters_is_an_empty_object_schema():
    spec = ToolSpec(name="noop", description="Does nothing.")
    assert spec.parameters == {"type": "object", "properties": {}}


# --- validate_arguments: structural checks -----------------------------------

def test_valid_arguments_pass():
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"table": "orders", "limit": 10})
    assert ok is True
    assert detail == "ok"


def test_missing_required_argument_rejected():
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"limit": 10})
    assert ok is False
    assert "table" in detail


def test_unknown_argument_rejected():
    """A tool call for a field that isn't in the schema must be rejected,
    not silently ignored -- an unexpected argument reaching a real tool
    later is exactly the kind of thing this layer exists to catch early."""
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"table": "orders", "drop_all": True})
    assert ok is False
    assert "drop_all" in detail


def test_wrong_type_rejected():
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"table": "orders", "limit": "ten"})
    assert ok is False
    assert "limit" in detail


def test_enum_violation_rejected():
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"table": "internal_secrets"})
    assert ok is False
    assert "table" in detail


def test_enum_valid_value_accepted():
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"table": "customers"})
    assert ok is True


def test_non_dict_arguments_rejected():
    spec = make_spec()
    ok, detail = validate_arguments(spec, ["not", "a", "dict"])
    assert ok is False
    assert "object" in detail


def test_bool_is_not_accepted_as_integer():
    """Python's bool is a subclass of int -- {"limit": true} must not
    silently pass an integer-typed field."""
    spec = make_spec()
    ok, detail = validate_arguments(spec, {"table": "orders", "limit": True})
    assert ok is False
    assert "limit" in detail


def test_bool_is_accepted_as_boolean():
    spec = make_spec(parameters={
        "type": "object",
        "properties": {"confirm": {"type": "boolean"}},
        "required": [],
    })
    ok, detail = validate_arguments(spec, {"confirm": True})
    assert ok is True


@pytest.mark.parametrize("json_type,good,bad", [
    ("string", "hello", 5),
    ("number", 3.14, "3.14"),
    ("array", [1, 2], "not a list"),
    ("object", {"a": 1}, "not a dict"),
])
def test_each_json_schema_type_is_checked(json_type, good, bad):
    spec = make_spec(parameters={
        "type": "object",
        "properties": {"field": {"type": json_type}},
        "required": ["field"],
    })
    assert validate_arguments(spec, {"field": good})[0] is True
    assert validate_arguments(spec, {"field": bad})[0] is False


def test_no_required_fields_means_empty_arguments_are_valid():
    spec = make_spec(parameters={
        "type": "object",
        "properties": {"optional_field": {"type": "string"}},
        "required": [],
    })
    ok, detail = validate_arguments(spec, {})
    assert ok is True


# --- check_tool_access: allow/deny by capability -----------------------------

def test_matching_capability_allowed():
    spec = make_spec(capability_required="ELEVATED")
    ok, detail = check_tool_access("ELEVATED", spec)
    assert ok is True


def test_higher_capability_allowed():
    """INTERNAL callers may use tools that only require ELEVATED -- rank
    is a minimum bar, not an exact-match requirement."""
    spec = make_spec(capability_required="ELEVATED")
    ok, detail = check_tool_access("INTERNAL", spec)
    assert ok is True


def test_lower_capability_denied():
    spec = make_spec(capability_required="INTERNAL")
    ok, detail = check_tool_access("GENERAL", spec)
    assert ok is False
    assert "INTERNAL" in detail
    assert "GENERAL" in detail


def test_general_tool_allows_every_tier():
    spec = make_spec(capability_required="GENERAL")
    for cap in ("GENERAL", "ELEVATED", "INTERNAL"):
        assert check_tool_access(cap, spec)[0] is True


def test_internal_tool_denies_general_and_elevated():
    spec = make_spec(capability_required="INTERNAL")
    assert check_tool_access("GENERAL", spec)[0] is False
    assert check_tool_access("ELEVATED", spec)[0] is False
    assert check_tool_access("INTERNAL", spec)[0] is True


def test_unrecognised_capability_denied_not_defaulted_to_lowest():
    """A caller capability that isn't one of the known tiers must be
    denied outright -- fail closed, never silently treated as GENERAL."""
    spec = make_spec(capability_required="GENERAL")
    ok, detail = check_tool_access("SUPERADMIN", spec)
    assert ok is False
    assert "unrecognised" in detail.lower()


# --- decide_tool_call: the combined pre-execution decision -------------------

def test_low_risk_call_with_valid_args_and_access_allows():
    spec = make_spec(risk_level="LOW")
    result = decide_tool_call("GENERAL", spec, {"table": "orders"})
    assert result["decision"] == "ALLOW"
    assert result["tool"] == "database.read"


def test_access_denial_blocks_before_validation_even_runs():
    """A caller who fails the capability check must be blocked even if
    their arguments would otherwise be perfectly valid -- access control
    is checked first, cheapest-and-most-decisive-first."""
    spec = make_spec(capability_required="INTERNAL", risk_level="LOW")
    result = decide_tool_call("GENERAL", spec, {"table": "orders"})
    assert result["decision"] == "BLOCK"
    assert "INTERNAL" in result["reason"]


def test_invalid_arguments_block_even_for_a_fully_authorized_caller():
    spec = make_spec(capability_required="GENERAL", risk_level="LOW")
    result = decide_tool_call("INTERNAL", spec, {})  # missing required "table"
    assert result["decision"] == "BLOCK"
    assert "table" in result["reason"]


def test_high_risk_requires_review_even_for_internal_caller():
    """The core property this item exists for: being ALLOWED to call a
    tool does not exempt a HIGH-risk call from human approval."""
    spec = make_spec(capability_required="GENERAL", risk_level="HIGH")
    result = decide_tool_call("INTERNAL", spec, {"table": "orders"})
    assert result["decision"] == "REVIEW"
    assert "HIGH" in result["reason"]


def test_high_risk_access_denial_still_blocks_not_review():
    """Access control outranks approval -- a caller who can't use the
    tool at all gets BLOCK, not REVIEW, regardless of the tool's risk
    level."""
    spec = make_spec(capability_required="INTERNAL", risk_level="HIGH")
    result = decide_tool_call("GENERAL", spec, {"table": "orders"})
    assert result["decision"] == "BLOCK"


def test_medium_risk_authorized_call_allows():
    spec = make_spec(risk_level="MEDIUM")
    result = decide_tool_call("GENERAL", spec, {"table": "orders"})
    assert result["decision"] == "ALLOW"


# --- ToolRegistry -------------------------------------------------------------

def test_register_and_get():
    reg = ToolRegistry()
    spec = make_spec()
    reg.register(spec)
    assert reg.get("database.read") is spec


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(make_spec())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(make_spec())


def test_get_unknown_tool_raises_keyerror():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="unknown tool"):
        reg.get("does_not_exist")


def test_list_tools_returns_all_registered():
    reg = ToolRegistry()
    reg.register(make_spec(name="tool_a"))
    reg.register(make_spec(name="tool_b"))
    names = {t.name for t in reg.list_tools()}
    assert names == {"tool_a", "tool_b"}


def test_len_and_contains():
    reg = ToolRegistry()
    assert len(reg) == 0
    assert "database.read" not in reg
    reg.register(make_spec())
    assert len(reg) == 1
    assert "database.read" in reg


def test_get_tool_registry_returns_a_shared_singleton():
    a = get_tool_registry()
    b = get_tool_registry()
    assert a is b
