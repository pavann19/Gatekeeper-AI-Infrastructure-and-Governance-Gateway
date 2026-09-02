"""
Fault-injection harness for Gatekeeper failure modes and safety invariant verification.

Verifies that when external dependencies fail (Redis, LLM judge backend, fusion models,
policy configuration files, or audit log storage), governance decisions fail CLOSED
(BLOCK/REVIEW/HIGH) and system availability gracefully degrades, rather than silently
failing open (false ALLOW) or dropping the audit trail.
"""
from __future__ import annotations

import unittest.mock as mock

from fastapi.testclient import TestClient

from api.main import app
from core import rate_limit as rl
from core import token_quota as tq
from core.circuit_breaker import ollama_judge_breaker
from core.config import settings
import core.fusion as fusion_mod
from core.output_guardrails import assess_output
from core.policy import FAIL_SAFE, _store, policy_decision, reload_policies
from core.risk import fuse_signals, judge_arbitration
from core.semantic_judge import output_judge

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Redis unreachable -> rate limiter + token quota fall back to local
# ---------------------------------------------------------------------------
def test_redis_unreachable_falls_back_to_local_and_assesses_safely():
    """Safety Property: When Redis is unreachable or raises connection errors,
    rate limiter and token quota trackers must fall back to local in-memory
    tracking so requests are still assessed without crashing or failing open.
    """
    fake_redis = mock.MagicMock()
    mock_script = mock.MagicMock()
    mock_script.side_effect = ConnectionError("Redis unreachable")
    fake_redis.register_script.return_value = mock_script
    fake_redis.get.side_effect = ConnectionError("Redis unreachable")

    # Instantiate Redis rate limiter and quota tracker with failing Redis
    limiter = rl.RedisRateLimiter(fake_redis, name="assess_test")
    tracker = tq.RedisTokenQuotaTracker(fake_redis)

    # 1. Rate limiter degrades gracefully to local in-memory bucket
    allowed, retry_after = limiter.check("test-tenant", capacity=5.0, refill_per_second=1.0)
    assert allowed is True
    assert retry_after == 0.0

    # 2. Token quota tracker degrades gracefully to local in-memory store
    assert tracker.would_exceed("test-tenant", 100) is False
    tracker.record("test-tenant", 50)
    assert tracker.usage_today("test-tenant") == 50

    # 3. Assess endpoint call with failing Redis completes and returns 200 (not crashing)
    with mock.patch("api.main.assess_rate_limiter", limiter), \
         mock.patch("api.main.gateway_token_quota", tracker), \
         mock.patch("api.main.assess_risk", return_value=("LOW", {"semantic_score": 0.0, "source": "clean_pass"})):
        response = client.post("/api/v1/assess", json={"prompt": "Hello world safe query"})
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] in ("ALLOW", "RESTRICT", "REVIEW", "BLOCK")


# ---------------------------------------------------------------------------
# 2. Judge backend unreachable -> fallback to toxic_bert, fail-closed if disabled
# ---------------------------------------------------------------------------
def test_judge_unreachable_fallback_and_fail_closed(monkeypatch):
    """Safety Property: When the primary judge backend is unreachable, the system
    must fall back to the local toxic_bert inverter. If the fallback is also
    disabled or unavailable, governance decisions must fail CLOSED (BLOCK/REVIEW / HIGH),
    never fail open (ALLOW / SAFE).
    """
    ollama_judge_breaker.reset()
    try:
        # Mock requests.post to simulate primary Ollama judge failure
        monkeypatch.setattr(
            "core.semantic_judge.requests.post",
            mock.MagicMock(side_effect=ConnectionError("Ollama down"))
        )

        # Case A: Fallback enabled -> output_judge engages toxic_bert inverter
        monkeypatch.setattr(settings, "OUTPUT_JUDGE_FALLBACK_ENABLED", True)
        toxic_text = "I hate you, you are a horrible disgusting piece of garbage and should die."
        verdict = output_judge(toxic_text)
        assert verdict == "DANGEROUS"

        decision, details = assess_output(toxic_text)
        assert decision == "BLOCK"
        assert details["toxicity_detected"] is True

        # Case B: Fallback disabled -> output_judge returns JUDGE_OFFLINE
        monkeypatch.setattr(settings, "OUTPUT_JUDGE_FALLBACK_ENABLED", False)
        verdict_disabled = output_judge(toxic_text)
        assert verdict_disabled == "JUDGE_OFFLINE"

        # Case C: In input judge arbitration, an unreachable judge must fail CLOSED to HIGH
        risk, source = judge_arbitration("Some ambiguous prompt")
        assert risk == "HIGH"
        assert source == "judge_failure_fail_closed"

    finally:
        ollama_judge_breaker.reset()


# ---------------------------------------------------------------------------
# 3. Fusion detector error -> degrades tier, never false ALLOW
# ---------------------------------------------------------------------------
def test_fusion_detector_failure_degrades_gracefully_never_false_allow(monkeypatch):
    """Safety Property: When a fusion detector raises an error during scoring,
    fused_threat_score must never fabricate or impute a zero/partial score.
    The ensemble must degrade to an available lower tier, or return available=False
    and fall back to deterministic safety signals rather than emitting a false ALLOW.
    """
    fusion_mod._load_policy()

    # Case A: An upgrade-tier detector fails (e.g. multilingual_head)
    def fake_score_detector(name, prompt):
        if name == "multilingual_head":
            return name, None, "RuntimeError: CUDA out of memory"
        return name, 0.2, None

    monkeypatch.setattr(fusion_mod, "_score_one_detector", fake_score_detector)
    res = fusion_mod.fused_threat_score("test prompt", anchor_score=0.1)

    # Should degrade to a lower reachable tier (e.g. eight_feature)
    assert res["available"] is True
    assert "eight_feature" in res.get("detail", "")

    # Case B: All detectors fail -> available=False, score is None (never 0.0)
    monkeypatch.setattr(
        fusion_mod,
        "_score_one_detector",
        lambda name, prompt: (name, None, "ModelCrashException")
    )
    res_all_fail = fusion_mod.fused_threat_score("test prompt", anchor_score=0.9)
    assert res_all_fail["available"] is False
    assert res_all_fail["score"] is None

    # Case C: fuse_signals with unavailable fusion and high anchor threat must NOT allow
    signals = {
        "meta_intent_score": 0.0,
        "domain_score": 0.5,
        "fusion_available": False,
        "fusion_score": None,
        "threat_score": 0.95,
        "is_educational": False,
        "centroid_score": 0.9,
    }
    risk, source, _judge_req, _topicality = fuse_signals(signals, "test prompt")
    assert risk == "HIGH"
    assert source == "vector_threat_critical"


# ---------------------------------------------------------------------------
# 4. policy_rules.json missing/malformed -> fails CLOSED to BLOCK
# ---------------------------------------------------------------------------
def test_policy_rules_missing_or_malformed_fails_closed(tmp_path):
    """Safety Property: When policy_rules.json is missing or malformed, the policy
    engine must fail CLOSED to BLOCK for every request and tenant, never guessing ALLOW.
    """
    original_path = _store.path
    try:
        # 1. Missing policy file -> fail closed
        missing_path = str(tmp_path / "nonexistent_policy.json")
        _store.path = missing_path
        reload_policies()
        assert _store._usable is False
        assert _store.get("default") == FAIL_SAFE

        action_gen_low, reason = policy_decision("GENERAL", "LOW", tenant_id="default")
        assert action_gen_low == "BLOCK"
        assert "Policies not loaded" in reason

        action_internal_low, _ = policy_decision("INTERNAL", "LOW", tenant_id="default")
        assert action_internal_low == "BLOCK"

        action_acme_elevated, _ = policy_decision("ELEVATED", "LOW", tenant_id="acme")
        assert action_acme_elevated == "BLOCK"

        # 2. Corrupt / malformed policy file -> fail closed
        corrupt_path = str(tmp_path / "corrupt_policy.json")
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content !!!")

        _store.path = corrupt_path
        reload_policies()
        assert _store._usable is False

        action_corrupt, _ = policy_decision("GENERAL", "LOW", tenant_id="default")
        assert action_corrupt == "BLOCK"

    finally:
        _store.path = original_path
        reload_policies()


# ---------------------------------------------------------------------------
# 5. Audit log path unwritable -> the request fails CLOSED, never a silent 200
# ---------------------------------------------------------------------------
def _raise_audit_error(*args, **kwargs):
    raise OSError("[Errno 28] No space left on device: 'audit.jsonl'")


def test_audit_log_unwritable_fails_closed_503(monkeypatch):
    """Safety Property: when the audit record cannot be persisted (disk full,
    permissions), the gateway must NOT return a normal 200 decision. It fails
    CLOSED with an explicit 503 that names the audit failure, so a verdict the
    system cannot prove it made never enters circulation.
    """
    monkeypatch.setattr(settings, "AUDIT_WRITE_FAILS_CLOSED", True)
    monkeypatch.setattr("api.main.log_event", _raise_audit_error)

    response = client.post("/api/v1/assess", json={"prompt": "Hello safe query"})
    assert response.status_code == 503
    assert "audit" in response.text.lower()
    assert response.headers.get("Retry-After") == "5"


def test_audit_log_unwritable_opt_out_serves_with_loud_log(monkeypatch, caplog):
    """An operator that has explicitly accepted un-audited traffic
    (AUDIT_WRITE_FAILS_CLOSED=false) still gets served, but the failure is
    logged at CRITICAL — it is never silent.
    """
    import logging
    monkeypatch.setattr(settings, "AUDIT_WRITE_FAILS_CLOSED", False)
    monkeypatch.setattr("api.main.log_event", _raise_audit_error)

    with caplog.at_level(logging.CRITICAL):
        response = client.post("/api/v1/assess", json={"prompt": "Hello safe query"})

    assert response.status_code == 200
    assert any("audit write failed" in r.message.lower() for r in caplog.records)
