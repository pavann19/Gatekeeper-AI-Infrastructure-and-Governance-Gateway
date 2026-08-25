"""
Tests for the Prometheus exporter and request correlation (§3.5).

Counters are process-global and cannot be reset between tests, so every
assertion here is a DELTA around the action under test rather than an
absolute value. Asserting absolutes would make these tests pass or fail
depending on what ran before them.
"""
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from api.main import app
from core import metrics
from core.auth import Principal
from core.logger import log_event

client = TestClient(app)

LOW_RISK = (
    "LOW",
    {
        "semantic_score": 0.1,
        "source": "fusion_clean_pass",
        "meta_intent_ms": 12.5,
        "faiss_threat_search_ms": 4.0,
        "fusion_ms": 250.0,
        "judge_invoked": False,
    },
)


def sample(name, **labels):
    """Current value of a metric sample, or 0.0 if it has no series yet."""
    value = REGISTRY.get_sample_value(name, labels or None)
    return 0.0 if value is None else value


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """These tests make many calls; limiting them is a different test's job."""
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


# ---------------------------------------------------------------------------
# The endpoint itself
# ---------------------------------------------------------------------------

def test_metrics_endpoint_serves_prometheus_exposition():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "gatekeeper_assessments_total" in response.text


def test_metrics_endpoint_can_be_disabled(monkeypatch):
    monkeypatch.setattr("api.main.settings.METRICS_ENABLED", False)
    assert client.get("/metrics").status_code == 404


def test_metrics_endpoint_can_require_authentication(monkeypatch):
    """
    Metrics leak operational detail — traffic volume, block rates, active
    tenants — so a deployment that cannot segment its network needs a gate.
    """
    monkeypatch.setattr("api.main.settings.METRICS_REQUIRE_AUTH", True)

    response = client.get("/metrics")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_metrics_endpoint_is_absent_from_the_public_schema():
    """It is an operational surface, not part of the API contract."""
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


# ---------------------------------------------------------------------------
# Assessment accounting
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_assessment_increments_the_outcome_counter(_mock):
    labels = dict(decision="ALLOW", risk_level="LOW", source="fusion_clean_pass")
    before = sample("gatekeeper_assessments_total", **labels)

    client.post("/api/v1/assess", json={"prompt": "hello"})

    assert sample("gatekeeper_assessments_total", **labels) == before + 1


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_stage_timings_are_exported_as_seconds(_mock):
    """
    The pipeline records these in MILLISECONDS and always has; Prometheus
    convention is base units. A missing conversion here would silently report
    every latency as 1000x too large, which no test of "did it record" would
    catch.
    """
    before_count = sample("gatekeeper_stage_duration_seconds_count", stage="fusion")
    before_sum = sample("gatekeeper_stage_duration_seconds_sum", stage="fusion")

    client.post("/api/v1/assess", json={"prompt": "hello"})

    assert sample("gatekeeper_stage_duration_seconds_count", stage="fusion") == before_count + 1
    observed = sample("gatekeeper_stage_duration_seconds_sum", stage="fusion") - before_sum
    assert observed == pytest.approx(0.250), "250ms must be exported as 0.25s"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_tenant_counter_uses_the_resolved_tenant(_mock):
    principal = Principal(
        capability="GENERAL", tenant="acme", key_id="k1",
        authenticated=True, reason="test",
    )
    before = sample("gatekeeper_tenant_assessments_total", tenant="acme", decision="ALLOW")

    with patch("api.main.resolve_principal", return_value=principal):
        client.post("/api/v1/assess", json={"prompt": "hello"})

    after = sample("gatekeeper_tenant_assessments_total", tenant="acme", decision="ALLOW")
    assert after == before + 1


@patch("api.main.assess_risk")
def test_cache_hit_without_stage_timings_still_records(mock_assess):
    """
    A cache hit carries no per-stage timings. Metrics must degrade, never
    raise — a KeyError here would turn an observability gap into a 500.
    """
    mock_assess.return_value = ("LOW", {"semantic_score": 0.1, "source": "cache"})
    before = sample(
        "gatekeeper_assessments_total",
        decision="ALLOW", risk_level="LOW", source="cache",
    )

    response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200
    after = sample(
        "gatekeeper_assessments_total",
        decision="ALLOW", risk_level="LOW", source="cache",
    )
    assert after == before + 1


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_instrumentation_failure_does_not_fail_the_request(_mock):
    """
    Observability must never be able to reject a request that was correctly
    assessed and decided.
    """
    with patch("api.main.metrics.record_assessment", side_effect=RuntimeError("boom")):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["risk_level"] == "LOW"


# ---------------------------------------------------------------------------
# Cardinality guards — the failure mode that takes the process down
# ---------------------------------------------------------------------------

def test_unknown_source_collapses_to_other_and_is_counted():
    """
    An unbounded label value is not a reporting bug, it is a memory leak. An
    unrecognised source must be bucketed, and the fact that it happened must
    be visible so the omission gets fixed.
    """
    before = sample("gatekeeper_metrics_unknown_source_total", source="brand_new_source")

    assert metrics.safe_source("brand_new_source") == "other"

    after = sample("gatekeeper_metrics_unknown_source_total", source="brand_new_source")
    assert after == before + 1


def test_known_sources_pass_through_unchanged():
    for source in ["cache", "fusion_threat_critical", "llama_guard_override"]:
        assert metrics.safe_source(source) == source


def test_every_source_risk_py_can_emit_is_known():
    """
    Guards against core/risk.py growing a verdict source that the exporter
    would silently bucket into 'other', quietly degrading the most useful
    label on the most useful metric.
    """
    import re
    from pathlib import Path

    text = Path("core/risk.py").read_text(encoding="utf-8")
    emitted = set(re.findall(r'return\s+"[A-Z]+",\s*"([a-z_]+)"', text))
    emitted |= set(re.findall(r'"source":\s*"([a-z_]+)"', text))
    emitted |= set(re.findall(r'source="([a-z_]+)"', text))

    unknown = emitted - metrics.KNOWN_SOURCES
    assert not unknown, f"core/risk.py emits sources the exporter doesn't know: {sorted(unknown)}"


def test_unmatched_paths_share_one_series():
    """
    Labelling by raw path would let anyone mint unlimited time series by
    requesting random URLs. 404s must all land in one bucket.
    """
    before = sample(
        "gatekeeper_request_duration_seconds_count",
        endpoint="unmatched", method="GET", status="404",
    )

    for i in range(5):
        client.get(f"/definitely-not-a-route-{i}")

    after = sample(
        "gatekeeper_request_duration_seconds_count",
        endpoint="unmatched", method="GET", status="404",
    )
    assert after == before + 5, "each unmatched path created its own series"


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_matched_routes_use_the_path_template(_mock):
    before = sample(
        "gatekeeper_request_duration_seconds_count",
        endpoint="/api/v1/assess", method="POST", status="200",
    )

    client.post("/api/v1/assess", json={"prompt": "hello"})

    after = sample(
        "gatekeeper_request_duration_seconds_count",
        endpoint="/api/v1/assess", method="POST", status="200",
    )
    assert after == before + 1


# ---------------------------------------------------------------------------
# Operational signals
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_rate_limited_requests_are_counted(_mock, monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ANONYMOUS_RPM", 60.0)
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_BURST_SECONDS", 2.0)

    before = sample("gatekeeper_rate_limited_total", authenticated="false")

    for _ in range(8):
        client.post("/api/v1/assess", json={"prompt": "hello"})

    assert sample("gatekeeper_rate_limited_total", authenticated="false") > before


def test_timeouts_are_counted(monkeypatch):
    import time as _time

    monkeypatch.setattr("api.main.settings.ASSESS_TIMEOUT_SECONDS", 0.05)
    before = sample("gatekeeper_assessment_timeouts_total", endpoint="/api/v1/assess")

    with patch("api.main.assess_risk", side_effect=lambda p, s, r=None: _time.sleep(1.0)):
        response = client.post("/api/v1/assess", json={"prompt": "hello"})

    assert response.status_code == 503
    after = sample("gatekeeper_assessment_timeouts_total", endpoint="/api/v1/assess")
    assert after == before + 1


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_in_flight_gauge_returns_to_zero(_mock):
    """A leaked increment would make the saturation signal useless over time."""
    client.post("/api/v1/assess", json={"prompt": "hello"})
    assert sample("gatekeeper_assessments_in_flight") == 0.0


def test_in_flight_gauge_is_decremented_even_on_timeout(monkeypatch):
    import time as _time

    monkeypatch.setattr("api.main.settings.ASSESS_TIMEOUT_SECONDS", 0.05)

    with patch("api.main.assess_risk", side_effect=lambda p, s, r=None: _time.sleep(0.5)):
        client.post("/api/v1/assess", json={"prompt": "hello"})

    assert sample("gatekeeper_assessments_in_flight") == 0.0


def test_scraping_does_not_mutate_circuit_breaker_state():
    """
    `is_open()` has a side effect: a cooled-down breaker transitions into its
    half-open probe when asked. If the exporter called it, a Prometheus scrape
    would be silently consuming probe attempts and changing failover timing —
    a monitor that alters the thing it measures.
    """
    from core.circuit_breaker import ollama_judge_breaker as breaker

    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    assert breaker._opened_at is not None

    # Age it past the cooldown, so is_open() WOULD transition it.
    breaker._opened_at -= breaker.cooldown_seconds + 1
    aged_value = breaker._opened_at

    metrics.refresh_circuit_breaker_gauges()

    assert breaker._opened_at == aged_value, "scraping tripped the half-open probe"
    assert sample("gatekeeper_circuit_breaker_open", backend="ollama_judge") == 1.0


# ---------------------------------------------------------------------------
# Request correlation
# ---------------------------------------------------------------------------

@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_request_id_is_generated_and_echoed(_mock):
    response = client.post("/api/v1/assess", json={"prompt": "hello"})

    request_id = response.headers.get("X-Request-ID")
    assert request_id
    assert len(request_id) >= 16


@patch("api.main.assess_risk", return_value=LOW_RISK)
def test_inbound_request_id_is_honoured(_mock):
    """A trace must be able to span services, not restart at this hop."""
    response = client.post(
        "/api/v1/assess",
        json={"prompt": "hello"},
        headers={"X-Request-ID": "upstream-trace-123"},
    )

    assert response.headers["X-Request-ID"] == "upstream-trace-123"


@patch("api.main.assess_risk", return_value=LOW_RISK)
@pytest.mark.parametrize("hostile", [
    "bad\nid",                       # newline: forges a second audit line
    "bad\r\nid",
    "id with spaces",
    "x" * 200,                       # unbounded length
    "tab\tid",
])
def test_malformed_request_ids_are_replaced_not_trusted(_mock, hostile):
    """
    THE INJECTION THIS PREVENTS: the ID lands in a JSONL audit log that
    downstream tooling parses line by line, so an unvalidated newline lets a
    caller forge additional audit records.
    """
    response = client.post(
        "/api/v1/assess",
        json={"prompt": "hello"},
        headers={"X-Request-ID": hostile},
    )

    returned = response.headers["X-Request-ID"]
    assert returned != hostile
    assert "\n" not in returned and "\r" not in returned
    assert len(returned) <= 64


def test_request_id_reaches_the_audit_record():
    """
    Without this, the only join key between a governance decision and the
    request that caused it is a timestamp — which stops being unique under
    any real concurrency.
    """
    records = []
    audit_logger = logging.getLogger("gatekeeper.audit")

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    audit_logger.addHandler(handler)
    try:
        log_event(
            "GENERAL", "some prompt", "LOW", "ALLOW",
            {"source": "cache", "request_id": "trace-abc-123"},
        )
    finally:
        audit_logger.removeHandler(handler)

    assert records, "no audit record was emitted"
    assert getattr(records[0], "request_id") == "trace-abc-123"


def test_audit_record_marks_a_missing_request_id_rather_than_omitting_it():
    """An absent field and an unset one are different; a query should see it."""
    records = []
    audit_logger = logging.getLogger("gatekeeper.audit")

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    audit_logger.addHandler(handler)
    try:
        log_event("GENERAL", "p", "LOW", "ALLOW", {"source": "cache"})
    finally:
        audit_logger.removeHandler(handler)

    assert getattr(records[0], "request_id") == "unset"
