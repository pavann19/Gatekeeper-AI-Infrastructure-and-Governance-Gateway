"""
Tests for the Stage 1.5 fast-path cascade (core/risk.py::_fast_path_signals /
_fast_path_decision) — the last unbuilt piece of the V2 reference
architecture's fast/deep split.

THE CLAIM UNDER TEST, stated precisely because it is easy to overstate: this
cascade can only ESCALATE to HIGH on a cheap, vector-only signal already
being confident. It cannot ALLOW, and it does not reduce latency for benign
or subtle-attack traffic — only for the subset of attacks a cheap anchor/
meta-intent similarity is already decisive about. See
docs/ENGINEERING_ASSESSMENT.md §1v for the measured before/after and this
scope boundary in full.
"""
from unittest.mock import patch

import pytest

from core import risk as risk_mod
from core.config import META_INTENT_THRESHOLD, SEMANTIC_THRESHOLD_HIGH
from core.risk import _fast_path_decision, _fast_path_signals


# ---------------------------------------------------------------------------
# _fast_path_decision: the pure decision logic, in isolation
# ---------------------------------------------------------------------------

def test_meta_intent_above_threshold_escalates():
    decision = _fast_path_decision({
        "meta_intent_score": META_INTENT_THRESHOLD + 0.01,
        "threat_score": 0.0,
    })
    assert decision == ("HIGH", "fast_path_meta_intent")


def test_meta_intent_at_exact_threshold_escalates():
    """>=, not >  — a score exactly AT the calibrated threshold must still
    escalate, matching the deep-path's own >= semantics (fuse_signals)."""
    decision = _fast_path_decision({
        "meta_intent_score": META_INTENT_THRESHOLD,
        "threat_score": 0.0,
    })
    assert decision is not None


def test_anchor_threat_above_threshold_escalates():
    decision = _fast_path_decision({
        "meta_intent_score": 0.0,
        "threat_score": SEMANTIC_THRESHOLD_HIGH + 0.01,
    })
    assert decision == ("HIGH", "fast_path_anchor_critical")


def test_neither_signal_decisive_returns_none():
    decision = _fast_path_decision({
        "meta_intent_score": META_INTENT_THRESHOLD - 0.2,
        "threat_score": SEMANTIC_THRESHOLD_HIGH - 0.2,
    })
    assert decision is None


def test_meta_intent_is_checked_before_anchor_threat():
    """When both are simultaneously decisive, the source must be
    deterministic, not whichever the implementation happens to check last."""
    decision = _fast_path_decision({
        "meta_intent_score": META_INTENT_THRESHOLD + 0.1,
        "threat_score": SEMANTIC_THRESHOLD_HIGH + 0.1,
    })
    assert decision == ("HIGH", "fast_path_meta_intent")


# ---------------------------------------------------------------------------
# _fast_path_signals: computed once, reused — no transformer calls
# ---------------------------------------------------------------------------

def test_fast_path_signals_never_touches_the_fusion_detectors():
    """
    THE ARCHITECTURAL CLAIM: these are vector-only operations. If this ever
    starts calling fused_threat_score (or anything that loads a transformer),
    the entire premise of the cascade — that this stage is cheap — is false.
    """
    with patch("core.risk.fused_threat_score") as mock_fusion:
        _fast_path_signals([0.0] * 768)
    assert mock_fusion.call_count == 0


def test_fast_path_signals_returns_both_keys():
    with patch("core.risk.check_meta_intent", return_value=0.3), \
         patch("core.risk.threat_store") as mock_store:
        mock_store.get_max_similarity.return_value = 0.4
        result = _fast_path_signals([0.0])

    assert result == {"meta_intent_score": 0.3, "threat_score": 0.4}


# ---------------------------------------------------------------------------
# End-to-end through assess_risk: the actual cascade, actually skipping fusion
# ---------------------------------------------------------------------------

@pytest.fixture
def drive_past_cache_and_hardban(monkeypatch):
    """Mocks only Stage 0 (cache) and Stage 1 (hard ban) so the fast-path
    stage and everything after it run for real."""
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))


def test_escalating_fast_path_skips_fusion_entirely(monkeypatch, drive_past_cache_and_hardban):
    """
    THE POINT OF THE CASCADE, proven end-to-end: when the fast path
    escalates, fused_threat_score — the expensive 3-transformer call — is
    NEVER invoked. This is the actual latency saving, not just a verdict
    matching what fusion would have said.
    """
    monkeypatch.setattr(risk_mod, "_fast_path_signals",
                        lambda vec: {"meta_intent_score": META_INTENT_THRESHOLD + 0.1,
                                     "threat_score": 0.0})

    with patch("core.risk.fused_threat_score") as mock_fusion:
        risk, details = risk_mod.assess_risk("irrelevant, fast path decides via the mock above")

    assert risk == "HIGH"
    assert details["source"] == "fast_path_meta_intent"
    assert mock_fusion.call_count == 0


def test_escalating_fast_path_skips_collect_semantic_signals_entirely(
    monkeypatch, drive_past_cache_and_hardban
):
    """Not just fusion — the WHOLE deep-signal-collection stage is skipped,
    including the cheap-but-still-unnecessary domain/centroid computations."""
    monkeypatch.setattr(risk_mod, "_fast_path_signals",
                        lambda vec: {"meta_intent_score": 0.0,
                                     "threat_score": SEMANTIC_THRESHOLD_HIGH + 0.1})

    with patch("core.risk.collect_semantic_signals") as mock_collect:
        risk, details = risk_mod.assess_risk("irrelevant")

    assert risk == "HIGH"
    assert details["source"] == "fast_path_anchor_critical"
    assert mock_collect.call_count == 0


def test_non_escalating_fast_path_falls_through_to_deep_path_unchanged(
    monkeypatch, drive_past_cache_and_hardban
):
    """
    ASYMMETRIC AUTHORITY, proven: a prompt that clears the fast path is in
    EXACTLY the same position as if the cascade did not exist — it still
    reaches collect_semantic_signals and fuse_signals. No ALLOW authority
    was granted to the cheap stage.
    """
    monkeypatch.setattr(risk_mod, "_fast_path_signals",
                        lambda vec: {"meta_intent_score": 0.0, "threat_score": 0.0})
    monkeypatch.setattr(risk_mod, "collect_semantic_signals", lambda p, v, fast=None: {
        "threat_score": 0.9, "dynamic_threat_score": 0.0, "is_educational": False,
        "domain_score": None, "domain_aligned": None, "meta_intent_score": 0.0,
        "centroid_score": 0.0, "fusion_available": False,
    })
    monkeypatch.setattr(risk_mod, "SEMANTIC_THRESHOLD_HIGH", 0.5)

    risk, details = risk_mod.assess_risk("clears the fast path, deep path still catches it")

    assert risk == "HIGH"
    assert details["source"] == "vector_threat_critical"  # the DEEP path's own source, not fast_path's


def test_fast_path_values_are_not_recomputed_in_stage_2(monkeypatch, drive_past_cache_and_hardban):
    """
    THE EFFICIENCY CLAIM: when the fast path does NOT escalate,
    collect_semantic_signals must reuse the already-computed meta_intent/
    threat_score rather than paying for the same two dot products again.
    """
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))

    with patch("core.risk.check_meta_intent", return_value=0.0) as mock_meta, \
         patch("core.risk.threat_store") as mock_store, \
         patch("core.risk.fused_threat_score",
               return_value={"available": False, "score": None, "threshold_high": None,
                            "threshold_medium": None, "detail": "test", "detector_scores": {}}), \
         patch("core.risk.compute_centroid_similarity", return_value=0.0), \
         patch("core.risk.check_educational_context", return_value=False):
        mock_store.get_max_similarity.return_value = 0.0
        risk_mod.assess_risk("clears the fast path, falls through to Stage 2")

    assert mock_meta.call_count == 1, "meta-intent recomputed in Stage 2 instead of reused"
    assert mock_store.get_max_similarity.call_count == 1, "threat_score recomputed in Stage 2 instead of reused"


# ---------------------------------------------------------------------------
# The decision source is registered for metrics — a silent 'other' bucket
# would defeat the whole point of adding a new, observable code path
# ---------------------------------------------------------------------------

def test_both_new_sources_are_registered_in_the_metrics_allowlist():
    from core.metrics import KNOWN_SOURCES
    assert "fast_path_meta_intent" in KNOWN_SOURCES
    assert "fast_path_anchor_critical" in KNOWN_SOURCES
