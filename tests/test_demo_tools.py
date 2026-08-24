"""
Tests for core/demo_tools.py — the first real tools run through
core/tools.py's full pipeline (access control, structural validation,
risk-based approval, execution), end to end rather than only unit-tested
against synthetic specs.

Each test uses a fresh ToolRegistry via register_demo_tools(registry=...)
rather than the process-wide shared one, so these tests can't leak state
into (or pick up state from) any other test module that touches the
shared registry.
"""
import pytest

from core.demo_tools import DEMO_TOOL_SPECS, register_demo_tools
from core.tools import ToolRegistry, execute_tool


@pytest.fixture
def registry():
    reg = ToolRegistry()
    register_demo_tools(reg)
    return reg


def test_all_four_demo_tools_registered(registry):
    names = {spec.name for spec in registry.list_tools()}
    assert names == {
        "demo.echo", "demo.calculator.add",
        "demo.database.query", "demo.database.delete",
    }


def test_register_demo_tools_is_idempotent(registry):
    """Calling it twice against the same registry must not raise --
    matches the module's own "not automatic, explicit opt-in" design:
    a caller might reasonably call this more than once across a
    process's lifetime (e.g. re-running setup in a test suite)."""
    register_demo_tools(registry)  # second call, same registry
    assert len(registry) == 4


def test_every_demo_tool_has_a_handler(registry):
    for spec in DEMO_TOOL_SPECS:
        assert registry.get_handler(spec.name) is not None


# --- demo.echo: the trivial ALLOW path ---------------------------------------

def test_echo_allows_and_returns_input(registry):
    result = execute_tool("GENERAL", "demo.echo", {"text": "hello"}, registry=registry)
    assert result["decision"] == "ALLOW"
    assert result["output"] == "hello"


def test_echo_rejects_missing_argument(registry):
    result = execute_tool("GENERAL", "demo.echo", {}, registry=registry)
    assert result["decision"] == "BLOCK"


# --- demo.calculator.add ------------------------------------------------------

def test_add_computes_correctly(registry):
    result = execute_tool("GENERAL", "demo.calculator.add", {"a": 2, "b": 3}, registry=registry)
    assert result["decision"] == "ALLOW"
    assert result["output"] == 5


def test_add_rejects_wrong_argument_type(registry):
    result = execute_tool("GENERAL", "demo.calculator.add",
                          {"a": "two", "b": 3}, registry=registry)
    assert result["decision"] == "BLOCK"


# --- demo.database.query: capability gate matters ----------------------------

def test_query_denied_for_general_capability(registry):
    """This tool requires ELEVATED -- a GENERAL caller must be blocked
    before the query ever touches the fake database."""
    result = execute_tool("GENERAL", "demo.database.query",
                          {"table": "orders"}, registry=registry)
    assert result["decision"] == "BLOCK"


def test_query_allowed_for_elevated_capability(registry):
    result = execute_tool("ELEVATED", "demo.database.query",
                          {"table": "orders"}, registry=registry)
    assert result["decision"] == "ALLOW"
    assert isinstance(result["output"], list)
    assert all("customer" in row for row in result["output"])


def test_query_allowed_for_internal_capability_too():
    """INTERNAL outranks ELEVATED -- the minimum-privilege rank check
    from the allow/deny item must apply here exactly as it does to any
    other tool."""
    reg = ToolRegistry()
    register_demo_tools(reg)
    result = execute_tool("INTERNAL", "demo.database.query", {"table": "customers"}, registry=reg)
    assert result["decision"] == "ALLOW"


def test_query_rejects_unknown_table(registry):
    result = execute_tool("ELEVATED", "demo.database.query",
                          {"table": "internal_secrets"}, registry=registry)
    assert result["decision"] == "BLOCK"


def test_query_respects_limit(registry):
    result = execute_tool("ELEVATED", "demo.database.query",
                          {"table": "orders", "limit": 1}, registry=registry)
    assert result["decision"] == "ALLOW"
    assert len(result["output"]) == 1


# --- demo.database.delete: HIGH risk forces REVIEW, even for INTERNAL -------

def test_delete_requires_review_even_for_internal_caller(registry):
    """The whole reason this tool exists at HIGH risk: being authorized
    to use it does not exempt the call from human approval."""
    result = execute_tool("INTERNAL", "demo.database.delete",
                          {"table": "orders", "row_id": 1}, registry=registry)
    assert result["decision"] == "REVIEW"
    assert "output" not in result  # never executed


def test_delete_denied_for_general_capability_before_review_is_even_considered(registry):
    """Access control outranks approval -- a GENERAL caller gets BLOCK,
    not REVIEW, since they couldn't use this tool at all regardless of
    its risk level."""
    result = execute_tool("GENERAL", "demo.database.delete",
                          {"table": "orders", "row_id": 1}, registry=registry)
    assert result["decision"] == "BLOCK"


def test_delete_never_actually_mutates_the_fake_database_via_review(registry):
    """A REVIEW decision must not be an execution in disguise -- confirm
    the row genuinely still exists afterward, not just that no exception
    was raised."""
    from core.demo_tools import _FAKE_DATABASE
    before = list(_FAKE_DATABASE["orders"])
    execute_tool("INTERNAL", "demo.database.delete", {"table": "orders", "row_id": 1}, registry=registry)
    assert _FAKE_DATABASE["orders"] == before


def test_delete_actually_removes_the_row_when_directly_invoked():
    """The handler itself works correctly -- exercised directly, since
    no capability can reach ALLOW for a HIGH-risk tool through
    execute_tool alone (REVIEW always intercepts it first). A real
    execution path for an approved REVIEW is a future item once the
    review queue is wired to tool calls (see core/tools.py's
    decide_tool_call docstring)."""
    from core.demo_tools import _database_delete, _FAKE_DATABASE
    original = list(_FAKE_DATABASE["orders"])
    try:
        _FAKE_DATABASE["orders"] = [{"id": 99, "customer": "test", "total": 1.0}]
        result = _database_delete("orders", 99)
        assert result["deleted"] == 1
        assert _FAKE_DATABASE["orders"] == []
    finally:
        # Module-level state -- restore it so this test can't affect
        # anything that runs after it in the same process.
        _FAKE_DATABASE["orders"] = original
