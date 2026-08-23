"""
Tests for the output audit event (Phase 2, Output Security).

THE GAP THIS CLOSES: /api/v1/assess_output previously called log_event
NOWHERE in its body -- a response could be BLOCKed for a leaked secret,
PII, toxicity, or hallucination with zero audit trail. These tests prove
both endpoints now emit a distinct output audit record via
core.logger.log_output_event, and that it carries a different shape than
the input record (event_type distinguishes them).
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

LOW_RISK = ("LOW", {"semantic_score": 0.1, "source": "mock"})
CLEAN_OUTPUT = ("ALLOW", {"source": "clean_pass", "clean_response": "a clean response"})
DIRTY_OUTPUT = ("BLOCK", {"source": "secret_leakage", "secrets_detected": True})


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


# ---------------------------------------------------------------------------
# /api/v1/assess_output previously logged nothing at all
# ---------------------------------------------------------------------------

def test_standalone_assess_output_now_emits_an_audit_event():
    with patch("core.output_guardrails.assess_output", return_value=CLEAN_OUTPUT), \
         patch("api.main.log_output_event") as mock_log:
        response = client.post("/api/v1/assess_output", json={"response_text": "hello"})

    assert response.status_code == 200
    mock_log.assert_called_once()


def test_standalone_assess_output_logs_even_on_block():
    """A BLOCKed response is exactly the case an auditor most needs a
    record of -- must not be the case that silently produces none."""
    with patch("core.output_guardrails.assess_output", return_value=DIRTY_OUTPUT), \
         patch("api.main.log_output_event") as mock_log:
        response = client.post("/api/v1/assess_output", json={"response_text": "leaky"})

    assert response.status_code == 200
    assert response.json()["decision"] == "BLOCK"
    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args
    # decision is a positional arg; confirm it's the BLOCK, not silently ALLOW.
    assert "BLOCK" in call_kwargs.args


def test_standalone_endpoint_passes_the_real_response_text_to_the_audit_call():
    with patch("core.output_guardrails.assess_output", return_value=CLEAN_OUTPUT), \
         patch("api.main.log_output_event") as mock_log:
        client.post("/api/v1/assess_output", json={"response_text": "the actual text"})

    assert "the actual text" in mock_log.call_args.args


# ---------------------------------------------------------------------------
# Combined /api/v1/assess: a DISTINCT output event, not just nested in the input one
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_combined_assess_emits_both_an_input_and_a_distinct_output_event(mock_assess):
    with patch("core.output_guardrails.assess_output", return_value=CLEAN_OUTPUT), \
         patch("api.main.log_event") as mock_input_log, \
         patch("api.main.log_output_event") as mock_output_log:
        response = client.post(
            "/api/v1/assess", json={"prompt": "hello", "response_text": "a response"},
        )

    assert response.status_code == 200
    mock_input_log.assert_called_once()
    mock_output_log.assert_called_once()


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_combined_assess_skips_the_output_event_when_no_response_text_given(mock_assess):
    with patch("api.main.log_event") as mock_input_log, \
         patch("api.main.log_output_event") as mock_output_log:
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200
    mock_input_log.assert_called_once()
    mock_output_log.assert_not_called()


# ---------------------------------------------------------------------------
# log_output_event itself: a genuinely distinct shape from log_event
# ---------------------------------------------------------------------------

def test_log_output_event_is_distinguishable_from_log_event_by_event_type():
    import logging
    from core.logger import log_event, log_output_event

    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.__dict__)

    audit_logger = logging.getLogger("gatekeeper.audit")
    handler = _Capture()
    audit_logger.addHandler(handler)
    try:
        log_event("GENERAL", "a prompt", "LOW", "ALLOW", {})
        log_output_event("GENERAL", "a response", "ALLOW", {})
    finally:
        audit_logger.removeHandler(handler)

    assert captured[0]["event_type"] == "input_assessment"
    assert captured[1]["event_type"] == "output_assessment"
    # The output record must never carry the raw response text, only its hash.
    assert "response_hash" in captured[1]
    assert "a response" not in str(captured[1])
