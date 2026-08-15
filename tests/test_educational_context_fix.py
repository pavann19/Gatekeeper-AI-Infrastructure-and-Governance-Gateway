"""
Regression tests for a real bug found while investigating the "dead dynamic
threat feed" Phase 1 item (docs/ROADMAP_V2.md; see
docs/ENGINEERING_ASSESSMENT.md section 1z).

THE BUG: `collect_semantic_signals` set `details["is_educational"]` by
calling `check_dynamic_safe_harbors(prompt_vec)` directly — a function that
only ever consults `DYNAMIC_SAFE_HARBORS`, a list nothing in the codebase
ever populated. It always returned 0.0 (falsy), so `signals["is_educational"]`
was always falsy regardless of how clearly a prompt was framed as authorized
security research — silently disabling the entire educational-safe-harbor
MEDIUM-downgrade path in `fuse_signals`. The correct function,
`check_educational_context`, already existed, was already correctly
implemented (checks the real, populated `educational_store` anchors against
a threshold), and was simply never called from the live signal-collection
path.

Existing tests (tests/test_fusion.py etc.) only ever construct a fake
`signals` dict with `is_educational` passed in directly, so they validate
`fuse_signals`'s reaction to the flag but never validate that the flag
itself gets computed correctly from a real prompt — which is exactly how
this bug survived. These tests close that gap.
"""
from unittest.mock import patch

from core.config import EDUCATIONAL_THRESHOLD
from core.risk import check_educational_context, collect_semantic_signals

# NOTE ON COVERAGE: this file's tests mock educational_store.get_max_similarity
# rather than embedding real text — CI's requirements-ci.txt deliberately
# excludes sentence-transformers (no ML model downloads for unit tests; see
# that file's comment). check_educational_context's behaviour against the
# REAL, populated educational_store was verified manually in development
# (genuine anchor-matching text -> True, unrelated text -> False) before this
# fix shipped — see docs/ENGINEERING_ASSESSMENT.md section 1z. What these
# tests guard in CI is the pure threshold logic and, more importantly, the
# WIRING (below) — that collect_semantic_signals actually calls this function
# rather than the dead one that caused the original bug.


# ---------------------------------------------------------------------------
# check_educational_context: threshold logic, against a mocked store score
# ---------------------------------------------------------------------------

def test_check_educational_context_true_above_threshold(monkeypatch):
    import core.risk as risk_mod
    monkeypatch.setattr(risk_mod.educational_store, "get_max_similarity",
                        lambda vec: EDUCATIONAL_THRESHOLD + 0.01)
    assert check_educational_context([0.0]) is True


def test_check_educational_context_false_below_threshold(monkeypatch):
    import core.risk as risk_mod
    monkeypatch.setattr(risk_mod.educational_store, "get_max_similarity",
                        lambda vec: EDUCATIONAL_THRESHOLD - 0.01)
    assert check_educational_context([0.0]) is False


# ---------------------------------------------------------------------------
# collect_semantic_signals: is_educational must come from the REAL check,
# not a function that can never return anything but a falsy 0.0
# ---------------------------------------------------------------------------

def test_signals_is_educational_reflects_check_educational_context(monkeypatch):
    """Proves the WIRING, not just the function: patch
    check_educational_context to a non-obvious sentinel and confirm
    `signals["is_educational"]` actually carries that value through,
    rather than being hardcoded or sourced from a different function."""
    import core.risk as risk_mod

    monkeypatch.setattr(risk_mod, "check_meta_intent", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod.threat_store, "get_max_similarity", lambda vec: 0.0)
    monkeypatch.setattr(risk_mod, "check_educational_context", lambda vec: True)
    monkeypatch.setattr(risk_mod, "is_domain_aligned", lambda p: (None, None))
    monkeypatch.setattr(risk_mod, "compute_centroid_similarity", lambda vec: 0.0)
    monkeypatch.setattr(
        risk_mod, "fused_threat_score",
        lambda prompt, anchor_score: {
            "available": False, "score": None, "threshold_high": None,
            "threshold_medium": None, "detail": "policy unavailable", "detector_scores": {},
        },
    )

    signals = risk_mod.collect_semantic_signals("irrelevant prompt", [0.0])
    assert signals["is_educational"] is True


def test_signals_never_calls_the_dead_dynamic_safe_harbor_function():
    """core.updates (the dead GitHub-fetch feed) was removed entirely --
    this guards against someone re-adding an equivalent always-empty
    dynamic-safe-harbor indirection without noticing it repeats the bug."""
    import core.risk as risk_mod
    assert not hasattr(risk_mod, "check_dynamic_safe_harbors")
    assert not hasattr(risk_mod, "check_dynamic_threats")


def test_core_updates_module_no_longer_exists():
    import importlib
    import pytest as _pytest
    with _pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.updates")


def test_details_no_longer_carries_dynamic_threat_score(monkeypatch):
    """Every dict shape assess_risk can return -- early-return paths and the
    deep-path Stage 5 return -- should no longer carry the removed field."""
    import core.risk as risk_mod

    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (True, "banned phrase"))

    risk, details = risk_mod.assess_risk("irrelevant")
    assert "dynamic_threat_score" not in details


def test_update_threat_intel_endpoint_removed():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.post("/api/v1/update")
    assert resp.status_code == 404
