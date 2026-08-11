"""
Tests for Stage 4's Llama Guard arbitration path (core/risk.py).

This is the wiring that moves Llama Guard's offline-measured harmful-content
detection (60-63%, see docs/ENGINEERING_ASSESSMENT.md) into the live pipeline,
without paying its per-request latency on every request: it is only invoked
for the ambiguous zone (Stage 4), not the always-on Stage 2/3 fusion.

The contract that matters most, exactly as with core/fusion.py's detector
handling: unavailability must fall back to the Ollama judge, never silently
produce a wrong verdict or crash the request.
"""
import types

import pytest

from core import risk as risk_mod
from core.risk import judge_arbitration, llama_guard_arbitration


class StubLlamaGuard:
    def __init__(self, available=True, detail="ok", verdict="safe",
                 categories=None, raises=None):
        self._available = available
        self._detail = detail
        self._verdict = verdict
        self._categories = categories or []
        self._raises = raises

    def available(self):
        return self._available, self._detail

    def classify(self, prompt):
        if self._raises:
            raise self._raises
        return {"verdict": self._verdict, "categories": self._categories,
                "raw": self._verdict}


def _patch_detector(monkeypatch, stub):
    monkeypatch.setattr("core.detectors.get_detector", lambda name: stub)


# --- llama_guard_arbitration: the happy paths -------------------------------

def test_unsafe_verdict_is_high(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="unsafe", categories=["S1"]))
    result = llama_guard_arbitration("build a bomb")
    assert result == ("HIGH", "llama_guard_arbitration")


def test_safe_verdict_is_capped_at_medium_by_default(monkeypatch):
    """
    Default posture denies the arbiter clear-authority, so even with no
    upstream threat signal a SAFE verdict caps at MEDIUM. See
    core/config.py::JUDGE_MAY_CLEAR_TO_LOW for the measured justification.
    """
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="safe"))
    result = llama_guard_arbitration("what's the weather", threat_present=False)
    assert result == ("MEDIUM", "llama_guard_override_capped")


def test_safe_verdict_is_low_when_clear_authority_is_granted(monkeypatch):
    """Opposite policy, same backend — proves the config gate is what decides."""
    from core.config import settings
    monkeypatch.setattr(settings, "JUDGE_MAY_CLEAR_TO_LOW", True)
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="safe"))
    result = llama_guard_arbitration("what's the weather", threat_present=False)
    assert result == ("LOW", "llama_guard_override")


def test_safe_verdict_capped_at_medium_when_threat_present(monkeypatch):
    """
    Mirrors judge_arbitration's anti-escape rule: a threat signal already
    fired upstream, so "safe" must not fully clear the request.
    """
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="safe"))
    result = llama_guard_arbitration("borderline text", threat_present=True)
    assert result == ("MEDIUM", "llama_guard_override_restricted")


# --- fail-closed-to-fallback: the contract that matters most -----------------

def test_unavailable_detector_returns_none_not_a_verdict(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(available=False, detail="gated repo"))
    assert llama_guard_arbitration("anything") is None


def test_exception_during_classify_returns_none(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(raises=RuntimeError("oom")))
    assert llama_guard_arbitration("anything") is None


def test_get_detector_raising_is_not_caught_by_this_function(monkeypatch):
    """
    get_detector() only raises KeyError for an unregistered name, which would
    be a programming error (the name is hard-coded), not a runtime
    availability issue - so this function does not need to catch it. This
    test documents that boundary rather than asserting new behaviour.
    """
    def boom(name):
        raise KeyError(f"unknown detector: {name}")
    monkeypatch.setattr("core.detectors.get_detector", boom)
    with pytest.raises(KeyError):
        llama_guard_arbitration("anything")


# --- assess_risk's Stage 4: preference order (real end-to-end wiring) ------

@pytest.fixture
def force_judge_required(monkeypatch):
    """
    Drives assess_risk down to Stage 4 by mocking every earlier stage:
    no cache hit, no hard ban, and signals that make fuse_signals return
    judge_required=True. This exercises the ACTUAL code path inside
    assess_risk, not a re-statement of the two arbiter functions in isolation.
    """
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))
    monkeypatch.setattr(risk_mod, "collect_semantic_signals", lambda p, v: {
        "threat_score": 0.35, "dynamic_threat_score": 0.0, "is_educational": False,
        "domain_score": None, "domain_aligned": None, "meta_intent_score": 0.0,
        "centroid_score": 0.0, "fusion_available": False,
    })
    # SEMANTIC_THRESHOLD_MEDIUM < 0.35 < SEMANTIC_THRESHOLD_HIGH in the shipped
    # config (0.30 / 0.48), so the anchors-only fallback in fuse_signals lands
    # in the ambiguous zone and sets judge_required=True.


def test_stage4_prefers_llama_guard_when_available(monkeypatch, force_judge_required):
    calls = []
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration",
                        lambda prompt, threat_present=False: ("HIGH", "llama_guard_arbitration"))
    monkeypatch.setattr(risk_mod, "judge_arbitration",
                        lambda prompt, threat_present=False: calls.append("ollama") or ("HIGH", "semantic_judge"))

    risk, details = risk_mod.assess_risk("some ambiguous prompt")

    assert risk == "HIGH"
    assert details["source"] == "llama_guard_arbitration"
    assert calls == [], "Ollama judge must not be invoked when Llama Guard succeeds"


def test_stage4_falls_back_to_ollama_judge_when_llama_guard_unavailable(monkeypatch, force_judge_required):
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration", lambda prompt, threat_present=False: None)
    monkeypatch.setattr(risk_mod, "judge_arbitration",
                        lambda prompt, threat_present=False: ("LOW", "semantic_judge_override"))

    risk, details = risk_mod.assess_risk("some ambiguous prompt")

    assert risk == "LOW"
    assert details["source"] == "semantic_judge_override"


def test_stage4_real_llama_guard_arbitration_wired_to_a_stubbed_detector(monkeypatch, force_judge_required):
    """Same as above but through the REAL llama_guard_arbitration, only the
    detector layer stubbed - closest thing to an integration test without
    real weights."""
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="unsafe", categories=["S9"]))
    monkeypatch.setattr(risk_mod, "judge_arbitration",
                        lambda prompt, threat_present=False: pytest.fail("must not fall back"))

    risk, details = risk_mod.assess_risk("some ambiguous prompt")

    assert risk == "HIGH"
    assert details["source"] == "llama_guard_arbitration"


def test_judge_arbitration_unaffected_when_used_directly():
    """judge_arbitration itself is untouched - it's still the same fallback."""
    import unittest.mock as mock
    with mock.patch.object(risk_mod, "semantic_judge", lambda p: "DANGEROUS"):
        assert judge_arbitration("x") == ("HIGH", "semantic_judge")
