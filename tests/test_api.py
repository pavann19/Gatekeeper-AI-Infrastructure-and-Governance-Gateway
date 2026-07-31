import pytest
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
