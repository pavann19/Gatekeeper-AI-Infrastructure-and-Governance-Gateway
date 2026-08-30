from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

@patch("api.main.assess_risk")
def test_assess_endpoint_clean(mock_assess_risk):
    # Mock assess_risk to return low risk so we don't trigger ML models in unit tests
    mock_assess_risk.return_value = ("LOW", {"semantic_score": 0.1, "source": "mock"})
    
    response = client.post("/api/v1/assess", json={"prompt": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "LOW"
    assert "decision" in data
    assert "clean_prompt" in data


@patch("api.main.assess_risk")
def test_assess_reports_topicality_separately(mock_assess_risk):
    """Topicality is a distinct field from risk_level, not folded into it."""
    mock_assess_risk.return_value = (
        "LOW",
        {"semantic_score": 0.1, "source": "clean_pass", "topicality": "OUT_OF_DOMAIN"},
    )

    response = client.post("/api/v1/assess", json={"prompt": "Best pasta recipe?"})
    assert response.status_code == 200
    data = response.json()
    # Off-topic must not inflate the safety verdict.
    assert data["risk_level"] == "LOW"
    assert data["topicality"] == "OUT_OF_DOMAIN"
    assert data["decision"] == "ALLOW"


@patch("api.main.assess_risk")
def test_assess_defaults_topicality_when_absent(mock_assess_risk):
    """A verdict carrying no topicality (e.g. cache hit) defaults to UNKNOWN."""
    mock_assess_risk.return_value = ("LOW", {"semantic_score": 0.1, "source": "cache"})

    response = client.post("/api/v1/assess", json={"prompt": "Hello"})
    assert response.status_code == 200
    assert response.json()["topicality"] == "UNKNOWN"


def test_health_check():
    """/health reports per-dependency status; overall status degrades if any fail.

    It cannot return a bare {"status": "healthy"} — the previous assertion here
    was unsatisfiable and had been failing on main.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in {"healthy", "degraded"}
    assert set(data["checks"]) == {
        "policy_files",
        "spacy_model",
        "embedding_model",
        "semantic_judge",
    }
    assert all(isinstance(v, bool) for v in data["checks"].values())
    # The overall status must be consistent with the individual checks.
    assert data["status"] == ("healthy" if all(data["checks"].values()) else "degraded")


# --- Phase 8 hardening: /health consults the circuit breaker instead of -----
# --- always making a fresh network call (found via a real load test showing
# --- /health latency badly degrading under concurrency when Ollama is down)

def test_health_skips_the_network_call_when_breaker_already_open():
    """A known-down backend should be reported instantly, with zero network
    round-trip -- not re-discovered the slow way on every single call."""
    from core.circuit_breaker import ollama_judge_breaker
    ollama_judge_breaker.reset()
    for _ in range(ollama_judge_breaker.failure_threshold):
        ollama_judge_breaker.record_failure()
    assert ollama_judge_breaker._opened_at is not None  # sanity: breaker is open

    try:
        with patch("requests.get") as mock_get:
            response = client.get("/health")
    finally:
        ollama_judge_breaker.reset()

    assert response.status_code == 200
    assert response.json()["checks"]["semantic_judge"] is False
    assert response.json()["status"] == "degraded"
    mock_get.assert_not_called()


def test_health_check_does_not_flip_the_breaker_half_open():
    """Reading breaker state for a health check must never consume or reset
    the one-shot half-open probe slot meant for a real judge call -- that
    would let /health silently report 'healthy' without ever verifying
    anything, and would mask a real ongoing outage from the actual judge
    path (which needs a fresh run of failures to re-trip after a probe)."""
    from core.circuit_breaker import ollama_judge_breaker
    ollama_judge_breaker.reset()
    for _ in range(ollama_judge_breaker.failure_threshold):
        ollama_judge_breaker.record_failure()
    opened_at_before = ollama_judge_breaker._opened_at
    failures_before = ollama_judge_breaker._consecutive_failures

    try:
        with patch("requests.get"):
            client.get("/health")
        assert ollama_judge_breaker._opened_at == opened_at_before
        assert ollama_judge_breaker._consecutive_failures == failures_before
    finally:
        ollama_judge_breaker.reset()


def test_health_still_probes_when_breaker_is_closed():
    """No cached negative signal yet -- /health should still make its
    existing real check rather than assuming healthy."""
    from core.circuit_breaker import ollama_judge_breaker
    ollama_judge_breaker.reset()

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        response = client.get("/health")

    mock_get.assert_called_once()
    assert response.json()["checks"]["semantic_judge"] is True


def test_health_check_fast_when_judge_unreachable():
    """When the judge backend is unreachable (and breaker closed), /health must:
    (a) return 200,
    (b) report semantic_judge: False,
    (c) complete promptly without blocking on long timeouts.
    """
    import time
    from core.circuit_breaker import ollama_judge_breaker
    ollama_judge_breaker.reset()

    t0 = time.perf_counter()
    response = client.get("/health")
    elapsed = time.perf_counter() - t0

    assert response.status_code == 200
    assert response.json()["checks"]["semantic_judge"] is False
    assert response.json()["status"] == "degraded"
    assert elapsed < 3.0, f"/health took {elapsed:.3f}s when judge was unreachable; must be < 3.0s"

