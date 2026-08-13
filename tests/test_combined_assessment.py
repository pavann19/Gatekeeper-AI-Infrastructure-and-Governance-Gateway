"""
Tests for combined input+output assessment on /api/v1/assess (Output Guard
wired into the main request loop — closes the V2 Phase 0 gap where a caller
had to remember to invoke /assess_output separately after generating a
response).

Gatekeeper does not call the caller's LLM itself — see api/schemas.py's
AssessRequest.response_text docstring. This tests the "submit both together"
pattern: caller assesses the prompt, generates their own response, and
submits both here in one call.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

LOW_RISK = ("LOW", {"semantic_score": 0.1, "source": "mock"})
MEDIUM_RISK = ("MEDIUM", {"semantic_score": 0.2, "source": "mock"})
HIGH_RISK = ("HIGH", {"semantic_score": 0.9, "source": "mock"})

CLEAN_OUTPUT = ("ALLOW", {"source": "clean_pass"})
DIRTY_OUTPUT = ("BLOCK", {"source": "pii_leakage", "pii_leakage": True})


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


# ---------------------------------------------------------------------------
# Backward compatibility: omitting response_text changes nothing
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_response_text_omitted_behaves_exactly_as_before(mock_assess):
    response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ALLOW"
    assert body["output_decision"] is None
    assert body["output_details"] is None


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_output_guardrails_not_invoked_when_response_text_omitted(mock_assess):
    with patch("core.output_guardrails.assess_output") as mock_output:
        client.post("/api/v1/assess", json={"prompt": "hello"})
    assert mock_output.call_count == 0


# ---------------------------------------------------------------------------
# The actual feature: combined assessment
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_clean_input_and_clean_output_both_allow(_mock):
    with patch("core.output_guardrails.assess_output", return_value=CLEAN_OUTPUT):
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "a clean reply"},
        )

    body = response.json()
    assert body["decision"] == "ALLOW"
    assert body["output_decision"] == "ALLOW"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_clean_input_but_dirty_output_escalates_to_block(_mock):
    """
    THE ACTUAL POINT of output guarding: an innocuous prompt does not
    guarantee a safe response. A clean input must not mask a leaky output.
    """
    with patch("core.output_guardrails.assess_output", return_value=DIRTY_OUTPUT):
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "leaky reply"},
        )

    body = response.json()
    assert body["decision"] == "BLOCK"
    assert body["output_decision"] == "BLOCK"
    assert body["details"]["output_assessment"]["pii_leakage"] is True


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_restrict_input_with_dirty_output_still_escalates_to_block(_mock):
    """BLOCK is more severe than RESTRICT — the combined decision must be
    the WORSE of the two, never the input's alone."""
    with patch("core.output_guardrails.assess_output", return_value=DIRTY_OUTPUT):
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "leaky reply"},
        )
    assert response.json()["decision"] == "BLOCK"


@patch("api.main.assess_risk", return_value=MEDIUM_RISK)
def test_restrict_input_with_clean_output_stays_restrict(_mock):
    """A clean output must not LOOSEN an input decision — ALLOW is less
    severe than RESTRICT and must never override it."""
    with patch("core.output_guardrails.assess_output", return_value=CLEAN_OUTPUT):
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "a clean reply"},
        )
    assert response.json()["decision"] == "RESTRICT"


# ---------------------------------------------------------------------------
# Already-blocked input skips output checking entirely
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=HIGH_RISK)
def test_blocked_input_skips_output_check(mock_assess):
    """
    Checking the output of a prompt that was never allowed through answers a
    question nobody asked — and wastes the one thing the bounded pool exists
    to protect.
    """
    with patch("core.output_guardrails.assess_output") as mock_output:
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "irrelevant"},
        )

    assert response.json()["decision"] == "BLOCK"
    assert mock_output.call_count == 0
    assert response.json()["output_decision"] is None


# ---------------------------------------------------------------------------
# Audit / metrics reflect the FINAL combined decision
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_audit_log_records_the_combined_decision_not_the_input_only_one(_mock):
    """
    A record showing ALLOW for a request that was actually blocked on its
    response would be a falsified audit trail — the exact failure mode this
    whole project is built to avoid.
    """
    import logging

    records = []
    audit_logger = logging.getLogger("gatekeeper.audit")

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    audit_logger.addHandler(handler)
    try:
        with patch("core.output_guardrails.assess_output", return_value=DIRTY_OUTPUT):
            client.post(
                "/api/v1/assess",
                json={"prompt": "hello", "response_text": "leaky reply"},
            )
    finally:
        audit_logger.removeHandler(handler)

    assert records
    assert records[-1].decision == "BLOCK"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_metrics_reflect_the_combined_decision(_mock):
    from prometheus_client import REGISTRY

    def sample(name, **labels):
        v = REGISTRY.get_sample_value(name, labels or None)
        return 0.0 if v is None else v

    before = sample(
        "gatekeeper_assessments_total", decision="BLOCK", risk_level="LOW", source="mock"
    )
    with patch("core.output_guardrails.assess_output", return_value=DIRTY_OUTPUT):
        client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "leaky reply"},
        )
    after = sample(
        "gatekeeper_assessments_total", decision="BLOCK", risk_level="LOW", source="mock"
    )
    assert after == before + 1


# ---------------------------------------------------------------------------
# Timeout on the output half
# ---------------------------------------------------------------------------

def test_output_timeout_fails_the_whole_request_not_a_fabricated_verdict(monkeypatch):
    """
    Same principle as the input-side timeout (§1q): a timeout is an
    availability event, not a security finding. The 503 must say plainly
    that NEITHER half was assessed.
    """
    import time as _time

    monkeypatch.setattr("api.main.settings.ASSESS_TIMEOUT_SECONDS", 0.05)

    with patch("api.main.assess_risk", return_value=LOW_RISK), \
         patch("core.output_guardrails.assess_output",
               side_effect=lambda t: _time.sleep(1.0)):
        response = client.post(
            "/api/v1/assess",
            json={"prompt": "hello", "response_text": "slow to check"},
        )

    assert response.status_code == 503
    assert "not" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Request size cap applies identically to response_text
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_oversized_response_text_is_rejected(mock_assess):
    response = client.post(
        "/api/v1/assess",
        json={"prompt": "hello", "response_text": "x" * 50_001},
    )
    assert response.status_code == 422
    assert mock_assess.call_count == 0
