"""
Endpoint-level enforcement of the §3.4 controls: rate limiting, request size
caps, and the assessment timeout.

These are deliberately API-level rather than unit tests. The controls only
matter if they fire on the real routes, in the real order — a limiter that
works perfectly but is checked after the expensive work has already been
scheduled protects nothing.
"""
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.auth import Principal

client = TestClient(app)

LOW_RISK = ("LOW", {"semantic_score": 0.1, "source": "mock"})


@pytest.fixture
def fast_limit(monkeypatch):
    """A tiny allowance so exhaustion takes a handful of requests, not 120."""
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ANONYMOUS_RPM", 60.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_AUTHENTICATED_RPM", 600.0)
    # 60/min = 1/sec; a 3-second burst window gives a 3-token bucket.
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_BURST_SECONDS", 3.0)


def authenticated_as(key_id):
    """Patches credential resolution to yield a verified principal."""
    return patch(
        "api.main.resolve_principal",
        return_value=Principal(
            capability="GENERAL",
            tenant="acme",
            key_id=key_id,
            authenticated=True,
            reason="test",
        ),
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_burst_is_served_then_limited_with_retry_after(_mock, fast_limit):
    codes = [
        client.post("/api/v1/assess", json={"prompt": "hello"}).status_code
        for _ in range(6)
    ]

    assert codes[:3] == [200, 200, 200], "the configured burst must be served"
    assert 429 in codes[3:], "sustained overage must be rejected"

    limited = client.post("/api/v1/assess", json={"prompt": "hello"})
    assert limited.status_code == 429
    # Without Retry-After a client has no signal but to hammer the endpoint,
    # which converts a rate limit into a busy-loop amplifier.
    assert int(limited.headers["Retry-After"]) >= 1


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_limiting_happens_before_the_expensive_work(mock_assess, fast_limit):
    """
    The whole point is to shed load, so a rejected request must never have
    reached the assessment pipeline.
    """
    for _ in range(3):
        client.post("/api/v1/assess", json={"prompt": "hello"})
    calls_before = mock_assess.call_count

    response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 429
    assert mock_assess.call_count == calls_before, "assess_risk ran despite the 429"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_separate_keys_do_not_share_a_budget(_mock, fast_limit):
    """One tenant exhausting its allowance must not deny service to another."""
    with authenticated_as("tenant-a"):
        for _ in range(30):
            client.post("/api/v1/assess", json={"prompt": "hello"})

    with authenticated_as("tenant-b"):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_authenticated_callers_get_the_larger_allowance(_mock, fast_limit):
    """
    Anonymous traffic is unattributable, so it is budgeted more tightly. At
    600/min vs 60/min, a verified key must still be served well past the point
    an anonymous caller would have been cut off.
    """
    for _ in range(4):
        client.post("/api/v1/assess", json={"prompt": "hello"})
    assert client.post("/api/v1/assess", json={"prompt": "hello"}).status_code == 429

    with authenticated_as("big-key"):
        codes = [
            client.post("/api/v1/assess", json={"prompt": "hello"}).status_code
            for _ in range(10)
        ]
    assert codes == [200] * 10


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_limiting_can_be_disabled(_mock, monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)

    codes = [
        client.post("/api/v1/assess", json={"prompt": "hello"}).status_code
        for _ in range(40)
    ]
    assert set(codes) == {200}


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_forwarded_for_is_ignored_unless_explicitly_trusted(_mock, fast_limit, monkeypatch):
    """
    THE BYPASS THIS PREVENTS: if X-Forwarded-For were trusted by default, any
    caller could rotate the header to mint a fresh bucket per request and
    ignore the limit entirely. Exhaust the budget, then try exactly that.
    """
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_TRUST_FORWARDED_FOR", False)

    for _ in range(4):
        client.post("/api/v1/assess", json={"prompt": "hello"})

    codes = [
        client.post(
            "/api/v1/assess",
            json={"prompt": "hello"},
            headers={"X-Forwarded-For": f"10.0.0.{i}"},
        ).status_code
        for i in range(5)
    ]
    assert set(codes) == {429}, "spoofed forwarded-for minted new budgets"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_trusted_forwarded_for_uses_the_rightmost_hop(_mock, fast_limit, monkeypatch):
    """
    With one trusted proxy, the rightmost entry is what the proxy observed;
    the leftmost is client-supplied. Taking the rightmost means a client that
    prepends a fake address still lands in its own real bucket.
    """
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_TRUST_FORWARDED_FOR", True)

    for i in range(4):
        client.post(
            "/api/v1/assess",
            json={"prompt": "hello"},
            headers={"X-Forwarded-For": f"1.2.3.{i}, 203.0.113.9"},
        )

    # Same real peer (rightmost), different claimed origin (leftmost).
    response = client.post(
        "/api/v1/assess",
        json={"prompt": "hello"},
        headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.9"},
    )
    assert response.status_code == 429


# ---------------------------------------------------------------------------
# Request size caps
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_oversized_prompt_is_rejected_before_assessment(mock_assess):
    response = client.post("/api/v1/assess", json={"prompt": "x" * 50_001})

    assert response.status_code == 422
    assert mock_assess.call_count == 0, "an oversized body reached the pipeline"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_empty_prompt_is_rejected(_mock):
    assert client.post("/api/v1/assess", json={"prompt": ""}).status_code == 422


def test_oversized_output_text_is_rejected():
    """
    This field previously had no cap at all, so it was the cheaper way to pin
    a worker: same expensive machinery, no length limit.
    """
    with patch("core.output_guardrails.assess_output") as mock_out:
        response = client.post(
            "/api/v1/assess_output", json={"response_text": "x" * 50_001}
        )

    assert response.status_code == 422
    assert mock_out.call_count == 0


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_slow_assessment_returns_503_not_a_fabricated_verdict(monkeypatch):
    """
    A timeout is an availability event, not a security finding. Returning a
    synthesised BLOCK would write a verdict into the audit log that no
    analysis produced — the failure mode §2 of the engineering assessment
    warns about, where infrastructure failure masquerades as detection signal.
    """
    monkeypatch.setattr("api.main.settings.ASSESS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)

    def slow_assess(prompt, scheduler, raw_prompt=None):
        time.sleep(1.0)
        return LOW_RISK

    with patch("api.main.assess_risk", side_effect=slow_assess):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 503
    assert response.headers.get("Retry-After") is not None
    # The caller must be told plainly that nothing was assessed, so a lenient
    # integration cannot mistake the error for an approval.
    assert "not" in response.json()["detail"].lower()


def test_fast_assessment_is_unaffected_by_the_timeout(monkeypatch):
    """The deadline must not fire on healthy traffic."""
    monkeypatch.setattr("api.main.settings.ASSESS_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)

    with patch("api.main.assess_risk", return_value=LOW_RISK):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["risk_level"] == "LOW"


# ---------------------------------------------------------------------------
# The output endpoint is no longer an unguarded side door
# ---------------------------------------------------------------------------

def test_output_endpoint_enforces_auth_mode(monkeypatch):
    """
    AUTH_MODE='required' documents that every request is attributed. This
    endpoint used to be exempt, which made that claim false and left a
    fully open path to the same expensive machinery.
    """
    monkeypatch.setattr("core.auth.settings.AUTH_MODE", "required")

    response = client.post("/api/v1/assess_output", json={"response_text": "hi"})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_output_endpoint_is_rate_limited(fast_limit):
    with patch("core.output_guardrails.assess_output", return_value=("ALLOW", {})):
        codes = [
            client.post("/api/v1/assess_output", json={"response_text": "hi"}).status_code
            for _ in range(6)
        ]

    assert 429 in codes, "the output endpoint bypassed rate limiting"
