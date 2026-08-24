"""
Tests for POST /api/v1/tools/call (Phase 6, Tool/Agent Gateway) -- the
endpoint that gives core/tools.py::execute_tool a real caller, real
tenant/request_id context, and a real REVIEW path via
core/review_queue.py.

Uses core/demo_tools.py's registered tools directly against the SHARED
registry (get_tool_registry()) rather than mocking execute_tool, so
these tests exercise the real access-control/validation/risk pipeline
end to end through the actual HTTP layer -- the same reasoning
tests/test_demo_tools.py gives for testing against real specs rather
than synthetic ones, one layer up.
"""
import json

import pytest
from fastapi.testclient import TestClient

import api.main as main_mod
import core.review_queue as review_queue_mod
from api.main import app
from core import auth as auth_mod
from core.auth import KeyStore, generate_key, hash_key
from core.demo_tools import register_demo_tools
from core.review_queue import ReviewQueue
from core.tools import get_tool_registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


@pytest.fixture(autouse=True)
def _demo_tools_registered():
    """Registers the demo tools into the SHARED registry for the
    duration of each test, then removes them -- these tests hit the real
    app/TestClient, which resolves tools via core.tools.get_tool_registry()
    (the module-level singleton), not an injectable one."""
    reg = register_demo_tools()
    yield reg
    for name in ("demo.echo", "demo.calculator.add", "demo.database.query", "demo.database.delete"):
        reg._tools.pop(name, None)
        reg._handlers.pop(name, None)


@pytest.fixture
def key_store(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    store = {}

    def issue(capability="GENERAL", tenant="default", key_id="test-key"):
        plaintext = generate_key()
        store[hash_key(plaintext)] = {"capability": capability, "tenant": tenant, "key_id": key_id}
        path.write_text(json.dumps(store), encoding="utf-8")
        monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
        return plaintext

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
    return issue


@pytest.fixture
def isolated_review_queue(tmp_path, monkeypatch):
    path = tmp_path / "review_queue.json"
    q = ReviewQueue(path=str(path))
    monkeypatch.setattr(review_queue_mod, "_queue", q)
    monkeypatch.setattr(main_mod, "enqueue_review", review_queue_mod.enqueue_review)
    monkeypatch.setattr(main_mod, "get_review", review_queue_mod.get_review)
    return q


# --- happy path: LOW-risk demo tool, anonymous (GENERAL) caller --------------

def test_echo_call_allows_and_returns_output():
    response = client.post("/api/v1/tools/call", json={"name": "demo.echo", "arguments": {"text": "hi"}})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ALLOW"
    assert body["tool"] == "demo.echo"
    assert body["output"] == "hi"
    assert body["error"] is None
    assert body["review_id"] is None


def test_calculator_add_computes_correctly():
    response = client.post("/api/v1/tools/call",
                           json={"name": "demo.calculator.add", "arguments": {"a": 2, "b": 3}})
    assert response.status_code == 200
    assert response.json()["output"] == 5


# --- access control: capability gate matters ---------------------------------

def test_general_caller_denied_medium_risk_elevated_tool():
    response = client.post("/api/v1/tools/call",
                           json={"name": "demo.database.query", "arguments": {"table": "orders"}})
    assert response.status_code == 200  # denial is a normal response, not an HTTP error
    body = response.json()
    assert body["decision"] == "BLOCK"
    assert body["output"] is None


def test_elevated_capability_allows_the_query(key_store):
    key = key_store(capability="ELEVATED")
    response = client.post(
        "/api/v1/tools/call",
        json={"name": "demo.database.query", "arguments": {"table": "orders"}},
        headers={"Authorization": f"Bearer {key}"},
    )
    body = response.json()
    assert body["decision"] == "ALLOW"
    assert isinstance(body["output"], list)


# --- structural validation ----------------------------------------------------

def test_missing_required_argument_blocks():
    response = client.post("/api/v1/tools/call", json={"name": "demo.echo", "arguments": {}})
    body = response.json()
    assert body["decision"] == "BLOCK"


def test_unknown_tool_blocks_cleanly():
    response = client.post("/api/v1/tools/call", json={"name": "does.not.exist", "arguments": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "BLOCK"
    assert "does.not.exist" in body["reason"]


# --- risk-based approval: HIGH risk enqueues a REAL review -------------------

def test_high_risk_call_enqueues_a_real_review(key_store, isolated_review_queue):
    """The core property this endpoint exists to close: a REVIEW decision
    is not just returned, it becomes a real, retrievable review record."""
    key = key_store(capability="INTERNAL")
    response = client.post(
        "/api/v1/tools/call",
        json={"name": "demo.database.delete", "arguments": {"table": "orders", "row_id": 1}},
        headers={"Authorization": f"Bearer {key}"},
    )
    body = response.json()
    assert body["decision"] == "REVIEW"
    assert body["review_id"] is not None
    assert body["output"] is None

    stored = isolated_review_queue.get(body["review_id"])
    assert stored is not None
    assert stored["status"] == "PENDING"
    assert stored["risk"] == "HIGH"
    assert stored["capability"] == "INTERNAL"


def test_review_enqueue_does_not_store_raw_arguments(key_store, isolated_review_queue):
    """Same privacy contract as every other reviewable item -- only an
    identifying hash, never the raw call content."""
    key = key_store(capability="INTERNAL")
    response = client.post(
        "/api/v1/tools/call",
        json={"name": "demo.database.delete", "arguments": {"table": "customers", "row_id": 999}},
        headers={"Authorization": f"Bearer {key}"},
    )
    review_id = response.json()["review_id"]
    stored = isolated_review_queue.get(review_id)
    dumped = json.dumps(stored)
    assert "999" not in dumped
    assert "customers" not in dumped
    assert "prompt_hash" in stored and len(stored["prompt_hash"]) == 64  # sha256 hex digest


def test_access_denial_never_enqueues_a_review(key_store, isolated_review_queue):
    """Access control outranks approval -- a caller who can't use the
    tool at all gets BLOCK, and BLOCK must never create a review."""
    response = client.post(
        "/api/v1/tools/call",
        json={"name": "demo.database.delete", "arguments": {"table": "orders", "row_id": 1}},
    )
    body = response.json()
    assert body["decision"] == "BLOCK"
    assert body["review_id"] is None
    assert isolated_review_queue.list_pending() == []


# --- handler failure vs security decision ------------------------------------

def test_handler_exception_reported_as_error_not_a_security_block():
    reg = get_tool_registry()
    from core.tools import ToolSpec

    def boom(**kwargs):
        raise RuntimeError("downstream unreachable")

    reg.register(ToolSpec(name="test.boom", description="Always fails.", risk_level="LOW"), handler=boom)
    try:
        response = client.post("/api/v1/tools/call", json={"name": "test.boom", "arguments": {}})
        body = response.json()
        assert body["decision"] == "ALLOW"  # the call WAS authorized to attempt running
        assert body["error"] is not None
        assert "RuntimeError" in body["error"]
    finally:
        del reg._tools["test.boom"]
        reg._handlers.pop("test.boom", None)


# --- auth boundary, same as every other endpoint -----------------------------

def test_requires_auth_when_auth_mode_required(monkeypatch):
    monkeypatch.setattr("api.main.settings.AUTH_MODE", "required")
    response = client.post("/api/v1/tools/call", json={"name": "demo.echo", "arguments": {"text": "hi"}})
    assert response.status_code == 401


def test_extra_fields_in_request_are_rejected():
    response = client.post("/api/v1/tools/call",
                           json={"name": "demo.echo", "arguments": {"text": "hi"}, "role": "INTERNAL"})
    assert response.status_code == 422
