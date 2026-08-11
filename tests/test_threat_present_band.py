"""
Tests for the threat_present fix (core/risk.py::_in_upper_ambiguous_band).

THE BUG: Stage 4 is only ever reached from fuse_signals' ambiguous-zone
branch (score >= threshold_medium), so re-checking `score >= threshold_medium`
at the Stage 4 call site — which is what `threat_present` used to be — was
tautologically True on every single invocation. A SAFE verdict from any
arbiter was therefore unconditionally capped at MEDIUM, and the Operational
benchmark metric could not move regardless of arbiter quality (this is
exactly what made the Llama Guard live-benchmark numbers look flat on that
metric — see docs/ENGINEERING_ASSESSMENT.md §1j and §1n).

These tests exercise the REAL call site through assess_risk end-to-end, with
only the underlying judge/detector mocked — not judge_arbitration or
llama_guard_arbitration themselves — because mocking those functions directly
(as the existing arbitration test suites do, for unrelated reasons) never
touches the threat_present computation this fix changes. A regression here
would have passed every pre-existing test in the suite silently.
"""
import unittest.mock as mock

import pytest

from core import risk as risk_mod
from core.risk import _in_upper_ambiguous_band

# Captured at import, before any fixture monkeypatches it away, so tests that
# need the REAL arbitration path can restore it.
_REAL_LLAMA_GUARD_ARBITRATION = risk_mod.llama_guard_arbitration


# --- _in_upper_ambiguous_band: the helper in isolation -----------------------

def test_lower_half_of_band_is_not_upper():
    # band is [0.10, 0.30); midpoint 0.20
    assert _in_upper_ambiguous_band(0.11, 0.10, 0.30) is False
    assert _in_upper_ambiguous_band(0.19, 0.10, 0.30) is False


def test_upper_half_of_band_is_upper():
    assert _in_upper_ambiguous_band(0.20, 0.10, 0.30) is True  # exact midpoint
    assert _in_upper_ambiguous_band(0.29, 0.10, 0.30) is True


def test_degenerate_band_fails_conservative():
    """threshold_high <= threshold_medium is a misconfiguration; must not
    divide by a non-positive band or silently return False."""
    assert _in_upper_ambiguous_band(0.5, 0.30, 0.30) is True
    assert _in_upper_ambiguous_band(0.5, 0.30, 0.10) is True


# --- end-to-end through assess_risk: the actual bug, actually fixed ---------

@pytest.fixture
def drive_to_stage4(monkeypatch):
    """
    Drives assess_risk to Stage 4 with a CONTROLLABLE fusion score, so a test
    can place it precisely in the lower or upper half of the ambiguous band.
    Only the judge backend (semantic_judge) is mocked — judge_arbitration and
    the Stage 4 call site run for real, which is the entire point.
    """
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))
    # Llama Guard unavailable -> forces the Ollama-judge path, whose
    # threat_present logic is what this bug affected.
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration", lambda *a, **k: None)

    def _set(score, threshold_medium=0.10, threshold_high=0.30):
        monkeypatch.setattr(risk_mod, "collect_semantic_signals", lambda p, v: {
            "threat_score": score, "dynamic_threat_score": 0.0, "is_educational": False,
            "domain_score": None, "domain_aligned": None, "meta_intent_score": 0.0,
            "centroid_score": 0.0, "fusion_available": True,
            "fusion_score": score, "fusion_threshold_medium": threshold_medium,
            "fusion_threshold_high": threshold_high,
        })
    return _set


def test_safe_verdict_in_lower_half_clears_to_low_only_when_permitted(monkeypatch, drive_to_stage4):
    """
    The band-split still works — but ONLY when the arbiter is explicitly
    granted clear-authority via JUDGE_MAY_CLEAR_TO_LOW.

    This test used to assert LOW unconditionally, because when the band-split
    landed (§1n) clearing was the new default. Measurement later reversed that
    default: see test_default_denies_the_arbiter_clear_authority below for the
    3.25:1 losing trade that drove the change.
    """
    monkeypatch.setattr(risk_mod.settings, "JUDGE_MAY_CLEAR_TO_LOW", True)
    drive_to_stage4(score=0.11, threshold_medium=0.10, threshold_high=0.30)  # lower half
    monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "SAFE")

    risk, details = risk_mod.assess_risk("mild ambiguous prompt")

    assert risk == "LOW"
    assert details["source"] == "semantic_judge_override"


def test_default_denies_the_arbiter_clear_authority(monkeypatch, drive_to_stage4):
    """
    THE DETERMINISTIC-ARBITER PROPERTY: with default config, an LLM's SAFE
    verdict cannot return a prompt the deterministic layer flagged to LOW.

    Measured justification, not preference. On the 546-prompt deepset
    benchmark the judge exercised clear-authority 17 times: 13 genuine attacks
    cleared, 4 genuine benign cleared — 3.25 true positives destroyed per
    false positive saved. Denying it moves recall 55.67% -> 62.07% and F1
    0.671 -> 0.712 for +1.17pp FPR.
    """
    drive_to_stage4(score=0.11, threshold_medium=0.10, threshold_high=0.30)  # lower half
    monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "SAFE")

    risk, details = risk_mod.assess_risk("mild ambiguous prompt")

    assert risk == "MEDIUM", "a SAFE verdict must not clear a flagged prompt by default"
    # Distinct from *_restricted so an audit can tell "score was high anyway"
    # apart from "policy denied the arbiter clear-authority".
    assert details["source"] == "semantic_judge_override_capped"


@pytest.fixture
def llama_guard_says_safe(monkeypatch):
    """
    Makes the REAL llama_guard_arbitration run, with only the multi-GB model
    replaced by a stub verdict. Stubbing the function itself would test
    nothing — the branch under test lives inside it.
    """
    stub = mock.Mock()
    stub.available.return_value = (True, "stubbed")
    stub.classify.return_value = {"verdict": "safe", "categories": ""}
    monkeypatch.setattr("core.detectors.get_detector", lambda name: stub)
    from core.circuit_breaker import llama_guard_breaker
    llama_guard_breaker.reset()
    return stub


def test_llama_guard_safe_is_also_capped_by_default(monkeypatch, drive_to_stage4,
                                                    llama_guard_says_safe):
    """The cap is a property of the arbitration stage, not of one backend."""
    drive_to_stage4(score=0.11, threshold_medium=0.10, threshold_high=0.30)
    # Undo the fixture's blanket disabling of Llama Guard so the real
    # arbitration path runs against the stubbed detector.
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration",
                        _REAL_LLAMA_GUARD_ARBITRATION)

    risk, details = risk_mod.assess_risk("mild ambiguous prompt")

    assert risk == "MEDIUM"
    assert details["source"] == "llama_guard_override_capped"


def test_llama_guard_safe_clears_when_permitted(monkeypatch, drive_to_stage4,
                                                llama_guard_says_safe):
    """Same backend, opposite policy — proves the gate is what decides."""
    monkeypatch.setattr(risk_mod.settings, "JUDGE_MAY_CLEAR_TO_LOW", True)
    drive_to_stage4(score=0.11, threshold_medium=0.10, threshold_high=0.30)
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration",
                        _REAL_LLAMA_GUARD_ARBITRATION)

    risk, details = risk_mod.assess_risk("mild ambiguous prompt")

    assert risk == "LOW"
    assert details["source"] == "llama_guard_override"


def test_safe_verdict_in_upper_half_still_restricted_to_medium(monkeypatch, drive_to_stage4):
    """
    The restriction is preserved exactly where it matters: a score close to
    the HIGH boundary getting a SAFE verdict is still the most consequential
    case for a fooled judge, and must still cap at MEDIUM.
    """
    drive_to_stage4(score=0.29, threshold_medium=0.10, threshold_high=0.30)  # upper half
    monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "SAFE")

    risk, details = risk_mod.assess_risk("severe ambiguous prompt")

    assert risk == "MEDIUM"
    assert details["source"] == "semantic_judge_override_restricted"


def test_dangerous_verdict_unaffected_by_band_position(monkeypatch, drive_to_stage4):
    """threat_present only gates the SAFE branch — DANGEROUS must return HIGH
    regardless of where in the band the score falls."""
    for score in (0.11, 0.29):
        drive_to_stage4(score=score, threshold_medium=0.10, threshold_high=0.30)
        monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "DANGEROUS")
        risk, details = risk_mod.assess_risk("prompt")
        assert risk == "HIGH"
        assert details["source"] == "semantic_judge"


def test_ambiguous_verdict_unaffected_by_band_position(monkeypatch, drive_to_stage4):
    for score in (0.11, 0.29):
        drive_to_stage4(score=score, threshold_medium=0.10, threshold_high=0.30)
        monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "AMBIGUOUS")
        risk, details = risk_mod.assess_risk("prompt")
        assert risk == "MEDIUM"
        assert details["source"] == "semantic_judge_ambiguous"


def test_anchors_only_fallback_path_gets_the_same_fix(monkeypatch):
    """The bug and the fix both existed identically in the anchors-only
    fallback branch (fusion_available=False) — must be covered too, not just
    the fusion path."""
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration", lambda *a, **k: None)
    # Clear-authority granted: this test is about the BAND SPLIT working on the
    # anchors-only path, which is only observable when clearing is permitted.
    monkeypatch.setattr(risk_mod.settings, "JUDGE_MAY_CLEAR_TO_LOW", True)
    monkeypatch.setattr(risk_mod, "SEMANTIC_THRESHOLD_MEDIUM", 0.10)
    monkeypatch.setattr(risk_mod, "SEMANTIC_THRESHOLD_HIGH", 0.30)
    monkeypatch.setattr(risk_mod, "collect_semantic_signals", lambda p, v: {
        "threat_score": 0.11, "dynamic_threat_score": 0.0, "is_educational": False,
        "domain_score": None, "domain_aligned": None, "meta_intent_score": 0.0,
        "centroid_score": 0.0, "fusion_available": False,
    })
    monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "SAFE")

    risk, details = risk_mod.assess_risk("mild ambiguous prompt, anchors-only path")

    assert risk == "LOW"
    assert details["source"] == "semantic_judge_override"


def test_pre_fix_behaviour_would_have_failed_this_test(monkeypatch, drive_to_stage4):
    """
    Documents the regression directly: simulating the OLD (buggy) computation
    inline shows it always evaluates True for any score that reached Stage 4,
    proving the fix changed real behaviour rather than being a no-op refactor.
    """
    monkeypatch.setattr(risk_mod.settings, "JUDGE_MAY_CLEAR_TO_LOW", True)
    drive_to_stage4(score=0.11, threshold_medium=0.10, threshold_high=0.30)
    old_threat_present = 0.11 >= 0.10  # the removed computation, literally
    assert old_threat_present is True  # tautological, as the bug's name says

    monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: "SAFE")
    risk, _ = risk_mod.assess_risk("mild ambiguous prompt")
    assert risk == "LOW"  # the NEW behaviour disagrees with the old computation
