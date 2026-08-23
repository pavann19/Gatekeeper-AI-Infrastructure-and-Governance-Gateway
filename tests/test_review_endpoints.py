"""
API-level tests for Phase 4 (Human Review): the REVIEW decision outcome,
its wiring into /api/v1/assess, and the three review endpoints.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core import policy as policy_mod
from core.auth import KeyStore, generate_key, hash_key
from core.policy import PolicyStore
from core.review_queue import ReviewQueue
import core.review_queue as review_queue_mod
import api.main as main_mod

client = TestClient(app)

LOW_RISK = ("LOW", {"semantic_score": 0.1, "source": "mock"})
MEDIUM_RISK = ("MEDIUM", {"semantic_score": 0.5, "source": "mock"})


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


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
def review_policy(tmp_path, monkeypatch):
    """A tenant policy that routes MEDIUM risk to REVIEW."""
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {
            "GENERAL": {"HIGH": "BLOCK", "MEDIUM": "REVIEW", "LOW": "ALLOW"},
        }}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))


@pytest.fixture
def isolated_review_queue(tmp_path, monkeypatch):
    path = tmp_path / "review_queue.json"
    q = ReviewQueue(path=str(path))
    monkeypatch.setattr(review_queue_mod, "_queue", q)
    monkeypatch.setattr(main_mod, "enqueue_review", review_queue_mod.enqueue_review)
    monkeypatch.setattr(main_mod, "get_review", review_queue_mod.get_review)
    monkeypatch.setattr(main_mod, "list_pending_reviews", review_queue_mod.list_pending_reviews)
    monkeypatch.setattr(main_mod, "resolve_review", review_queue_mod.resolve_review)
    return q


# ---------------------------------------------------------------------------
# REVIEW decision wiring on /api/v1/assess
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_medium_risk_under_review_policy_returns_review_decision(mock_assess, review_policy, isolated_review_queue):
    response = client.post("/api/v1/assess", json={"prompt": "an ambiguous prompt"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "REVIEW"
    assert body["review_id"] is not None


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_review_id_is_none_for_non_review_decisions(mock_assess, isolated_review_queue):
    """Default policy (no REVIEW mapping) -- review_id must stay None."""
    response = client.post("/api/v1/assess", json={"prompt": "hello"})
    assert response.json()["review_id"] is None


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_review_enqueues_without_storing_raw_prompt(mock_assess, review_policy, isolated_review_queue):
    response = client.post("/api/v1/assess", json={"prompt": "a very specific secret-shaped prompt"})
    review_id = response.json()["review_id"]

    record = isolated_review_queue.get(review_id)
    assert "a very specific secret-shaped prompt" not in json.dumps(record)
    assert "prompt_hash" in record


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_output_guard_is_skipped_when_input_decision_is_review(mock_assess, review_policy, isolated_review_queue):
    """Checking the output of a prompt that isn't allowed through YET
    answers a question nobody asked -- same reasoning as skipping it for BLOCK."""
    with patch("core.output_guardrails.assess_output") as mock_output:
        response = client.post(
            "/api/v1/assess", json={"prompt": "ambiguous", "response_text": "some response"},
        )
        mock_output.assert_not_called()
    assert response.json()["decision"] == "REVIEW"


@patch("api.main.assess_risk", return_value=("HIGH", {"semantic_score": 0.9, "source": "mock"}))
def test_review_does_not_override_a_more_severe_block(mock_assess, isolated_review_queue, tmp_path, monkeypatch):
    """BLOCK > REVIEW in severity -- a HIGH-risk input that's already BLOCK
    stays BLOCK even if output somehow (misconfigured) evaluated to REVIEW-shaped."""
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    response = client.post("/api/v1/assess", json={"prompt": "dangerous"})
    assert response.json()["decision"] == "BLOCK"
    assert response.json()["review_id"] is None


# ---------------------------------------------------------------------------
# GET /api/v1/review/{review_id}
# ---------------------------------------------------------------------------

def test_get_review_status_pending(isolated_review_queue):
    record = isolated_review_queue.enqueue(
        reason="r", capability="GENERAL", risk="MEDIUM", tenant="default",
        prompt_hash="h", request_id="req-1",
    )
    response = client.get(f"/api/v1/review/{record.review_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PENDING"
    assert body["final_decision"] is None


def test_get_review_status_unknown_id_404(isolated_review_queue):
    response = client.get("/api/v1/review/nonexistent-id")
    assert response.status_code == 404


def test_get_review_status_requires_auth_when_required(isolated_review_queue, monkeypatch):
    monkeypatch.setattr("api.main.settings.AUTH_MODE", "required")
    record = isolated_review_queue.enqueue(
        reason="r", capability="GENERAL", risk="MEDIUM", tenant="default",
        prompt_hash="h", request_id="req-1",
    )
    response = client.get(f"/api/v1/review/{record.review_id}")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/review (list pending) -- INTERNAL only
# ---------------------------------------------------------------------------

def test_list_reviews_requires_internal_capability(isolated_review_queue, key_store):
    key = key_store(capability="GENERAL")
    response = client.get("/api/v1/review", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_list_reviews_rejects_anonymous(isolated_review_queue):
    response = client.get("/api/v1/review")
    assert response.status_code == 403


def test_list_reviews_succeeds_for_internal_capability(isolated_review_queue, key_store):
    isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                                  tenant="default", prompt_hash="h", request_id="req-1")
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/review", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert len(response.json()["pending"]) == 1


def test_list_reviews_excludes_resolved(isolated_review_queue, key_store):
    r = isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                                      tenant="default", prompt_hash="h", request_id="req-1")
    isolated_review_queue.resolve(r.review_id, "APPROVED", reviewer="someone")
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/review", headers={"Authorization": f"Bearer {key}"})
    assert response.json()["pending"] == []


# ---------------------------------------------------------------------------
# POST /api/v1/review/{review_id}/resolve -- INTERNAL only
# ---------------------------------------------------------------------------

def test_resolve_requires_internal_capability(isolated_review_queue, key_store):
    record = isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                                           tenant="default", prompt_hash="h", request_id="req-1")
    key = key_store(capability="ELEVATED")
    response = client.post(
        f"/api/v1/review/{record.review_id}/resolve",
        json={"outcome": "APPROVED"}, headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 403


def test_resolve_approve_feeds_back_as_allow(isolated_review_queue, key_store):
    record = isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                                           tenant="default", prompt_hash="h", request_id="req-1")
    key = key_store(capability="INTERNAL", key_id="reviewer-alice")
    response = client.post(
        f"/api/v1/review/{record.review_id}/resolve",
        json={"outcome": "APPROVED"}, headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["final_decision"] == "ALLOW"
    assert body["reviewer"] == "reviewer-alice"


def test_resolve_reject_feeds_back_as_block(isolated_review_queue, key_store):
    record = isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="HIGH",
                                           tenant="default", prompt_hash="h", request_id="req-1")
    key = key_store(capability="INTERNAL")
    response = client.post(
        f"/api/v1/review/{record.review_id}/resolve",
        json={"outcome": "REJECTED"}, headers={"Authorization": f"Bearer {key}"},
    )
    assert response.json()["final_decision"] == "BLOCK"


def test_resolve_invalid_outcome_value_is_422(isolated_review_queue, key_store):
    record = isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                                           tenant="default", prompt_hash="h", request_id="req-1")
    key = key_store(capability="INTERNAL")
    response = client.post(
        f"/api/v1/review/{record.review_id}/resolve",
        json={"outcome": "MAYBE"}, headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 422


def test_resolve_unknown_review_is_404(isolated_review_queue, key_store):
    key = key_store(capability="INTERNAL")
    response = client.post(
        "/api/v1/review/nonexistent/resolve",
        json={"outcome": "APPROVED"}, headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 404


def test_resolve_twice_is_409(isolated_review_queue, key_store):
    record = isolated_review_queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                                           tenant="default", prompt_hash="h", request_id="req-1")
    key = key_store(capability="INTERNAL")
    client.post(f"/api/v1/review/{record.review_id}/resolve",
               json={"outcome": "APPROVED"}, headers={"Authorization": f"Bearer {key}"})
    second = client.post(f"/api/v1/review/{record.review_id}/resolve",
                         json={"outcome": "REJECTED"}, headers={"Authorization": f"Bearer {key}"})
    assert second.status_code == 409
