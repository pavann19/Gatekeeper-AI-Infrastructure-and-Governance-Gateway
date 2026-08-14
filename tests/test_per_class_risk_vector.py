"""
Tests for surfacing the per-class risk vector (Gatekeeper 2.0 Phase 1 —
see docs/ROADMAP_V2.md).

core/fusion.py's `fused_threat_score` already computes `triggering_class`
and `class_scores` (via `_select_per_class_verdict`) but core/risk.py
previously dropped both before they reached `details` — the caller could
see THAT a request was flagged HIGH, never WHAT KIND of attack fusion
believed it was. These tests prove the two fields survive every path that
builds a `details`/return dict in core/risk.py, including the early-return
paths (cache/hard-ban/fast-path) where they must be present but empty.
"""
from unittest.mock import patch

import pytest

from core import risk as risk_mod
from core.config import META_INTENT_THRESHOLD


# ---------------------------------------------------------------------------
# collect_semantic_signals: the fields must be copied from fused_threat_score
# ---------------------------------------------------------------------------

def test_collect_semantic_signals_copies_triggering_class_and_class_scores(monkeypatch):
    monkeypatch.setattr(risk_mod, "check_meta_intent", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod.threat_store, "get_max_similarity", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod, "check_dynamic_threats", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod, "check_dynamic_safe_harbors", lambda vec: False)
    monkeypatch.setattr(risk_mod, "is_domain_aligned", lambda p: (None, None))
    monkeypatch.setattr(risk_mod, "compute_centroid_similarity", lambda vec: 0.0)
    monkeypatch.setattr(
        risk_mod, "fused_threat_score",
        lambda prompt, anchor_score: {
            "available": True, "score": 0.91, "threshold_high": 0.8,
            "threshold_medium": 0.5, "detail": "fusion applied (per-class, triggered by jailbreak)",
            "detector_scores": {}, "triggering_class": "jailbreak",
            "class_scores": {"jailbreak": 0.91, "prompt_injection": 0.2, "harmful_content": 0.05},
        },
    )

    signals = risk_mod.collect_semantic_signals("irrelevant prompt", [0.0])

    assert signals["fusion_triggering_class"] == "jailbreak"
    assert signals["fusion_class_scores"] == {
        "jailbreak": 0.91, "prompt_injection": 0.2, "harmful_content": 0.05,
    }


def test_collect_semantic_signals_defaults_when_fusion_has_no_per_class_result(monkeypatch):
    """FUSION_PER_CLASS off, or a v1 policy artifact with no per_class
    section — fused_threat_score returns triggering_class=None,
    class_scores={} (core/fusion.py:409-410), and that must pass through
    unchanged rather than being defaulted to something misleading."""
    monkeypatch.setattr(risk_mod, "check_meta_intent", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod.threat_store, "get_max_similarity", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod, "check_dynamic_threats", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod, "check_dynamic_safe_harbors", lambda vec: False)
    monkeypatch.setattr(risk_mod, "is_domain_aligned", lambda p: (None, None))
    monkeypatch.setattr(risk_mod, "compute_centroid_similarity", lambda vec: 0.0)
    monkeypatch.setattr(
        risk_mod, "fused_threat_score",
        lambda prompt, anchor_score: {
            "available": True, "score": 0.3, "threshold_high": 0.8,
            "threshold_medium": 0.5, "detail": "fusion applied",
            "detector_scores": {}, "triggering_class": None, "class_scores": {},
        },
    )

    signals = risk_mod.collect_semantic_signals("irrelevant prompt", [0.0])

    assert signals["fusion_triggering_class"] is None
    assert signals["fusion_class_scores"] == {}


# ---------------------------------------------------------------------------
# assess_risk end-to-end: the fields must reach the final returned dict
# ---------------------------------------------------------------------------

@pytest.fixture
def drive_past_cache_and_hardban(monkeypatch):
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))
    # Non-decisive fast path so execution reaches collect_semantic_signals.
    monkeypatch.setattr(risk_mod, "_fast_path_signals",
                        lambda vec: {"meta_intent_score": 0.0, "threat_score": 0.0})


def test_assess_risk_deep_path_surfaces_the_class_vector(monkeypatch, drive_past_cache_and_hardban):
    monkeypatch.setattr(
        risk_mod, "collect_semantic_signals",
        lambda p, v, fast=None: {
            "meta_intent_score": 0.0, "threat_score": 0.0,
            "dynamic_threat_score": 0.0, "is_educational": False,
            "domain_aligned": None, "domain_score": None, "centroid_score": 0.0,
            "fusion_available": True, "fusion_score": 0.91,
            "fusion_threshold_high": 0.8, "fusion_threshold_medium": 0.5,
            "fusion_detail": "fusion applied (per-class, triggered by harmful_content)",
            "fusion_detector_scores": {},
            "fusion_triggering_class": "harmful_content",
            "fusion_class_scores": {"harmful_content": 0.91, "jailbreak": 0.1, "prompt_injection": 0.05},
        },
    )
    monkeypatch.setattr(risk_mod, "fuse_signals", lambda signals, prompt: ("HIGH", "fusion", False, "UNKNOWN"))

    risk, details = risk_mod.assess_risk("irrelevant, mocked above")

    assert risk == "HIGH"
    assert details["fusion_triggering_class"] == "harmful_content"
    assert details["fusion_class_scores"]["harmful_content"] == 0.91


@pytest.mark.parametrize("path_name,setup", [
    ("cache_locked_high", lambda monkeypatch: monkeypatch.setattr(
        risk_mod, "lookup_cache", lambda prompt, vec: ("HIGH", 1.0))),
    ("symbolic_rule", lambda monkeypatch: monkeypatch.setattr(
        risk_mod, "hard_ban_triggered", lambda p: (True, "banned phrase"))),
])
def test_early_return_paths_include_empty_class_vector_keys(monkeypatch, path_name, setup):
    """Cache-hit and hard-ban short-circuits never touch fusion, but callers
    reading `details["fusion_class_scores"]` unconditionally must not KeyError
    just because a request happened to short-circuit."""
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))
    setup(monkeypatch)

    risk, details = risk_mod.assess_risk("irrelevant")

    assert details["fusion_triggering_class"] is None
    assert details["fusion_class_scores"] == {}


def test_escalating_fast_path_includes_empty_class_vector_keys(monkeypatch, drive_past_cache_and_hardban):
    """The fast-path escalation (§1v) never reaches fusion either — same
    contract as the cache/hard-ban paths."""
    monkeypatch.setattr(risk_mod, "_fast_path_signals",
                        lambda vec: {"meta_intent_score": META_INTENT_THRESHOLD + 0.1,
                                     "threat_score": 0.0})

    with patch("core.risk.fused_threat_score"):
        risk, details = risk_mod.assess_risk("irrelevant, fast path decides")

    assert risk == "HIGH"
    assert details["fusion_triggering_class"] is None
    assert details["fusion_class_scores"] == {}
