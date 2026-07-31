"""
Tests for core/fusion.py — the runtime application of the trained multi-
detector policy.

The critical property under test is FAIL-CLOSED-TO-FALLBACK, not fail-open and
not fail-crash: if any live detector is unavailable, `fused_threat_score` must
report `available: False` rather than silently imputing a score for the
missing feature (which would understate risk in exactly the situation where a
detector has gone dark) or raising (which would take the whole gateway down).
core/risk.py depends on this contract to fall back to the anchors-only path.
"""
import json

import pytest

from core import fusion as fusion_mod


ARTIFACT = {
    "version": 1,
    "trained_at": "2026-01-01T00:00:00+00:00",
    "feature_order": ["anchors", "protectai_injection", "madhurjindal_jailbreak", "toxic_bert"],
    "scaler_mean": [0.2, 0.1, 0.1, 0.05],
    "scaler_scale": [0.25, 0.2, 0.2, 0.1],
    "coefficients": [1.5, 1.6, 0.6, 0.5],
    "intercept": -1.2,
    "threshold_high": 0.28,
    "threshold_medium": 0.10,
    "fpr_budget_high": 0.05,
    "fpr_budget_medium": 0.20,
    "training": {"n_rows": 6933},
}


@pytest.fixture(autouse=True)
def reset_policy_cache():
    """Each test gets a clean module-level cache — these globals are lazy-loaded once."""
    fusion_mod._policy = None
    fusion_mod._policy_error = None
    fusion_mod._policy_loaded = False
    yield
    fusion_mod._policy = None
    fusion_mod._policy_error = None
    fusion_mod._policy_loaded = False


@pytest.fixture
def artifact_file(tmp_path, monkeypatch):
    path = tmp_path / "fusion_policy.json"
    path.write_text(json.dumps(ARTIFACT), encoding="utf-8")
    monkeypatch.setattr(fusion_mod, "ARTIFACT_FILE", str(path))
    return path


class StubDetector:
    def __init__(self, score=None, available=True, detail="ok", raises=None):
        self._score = score
        self._available = available
        self._detail = detail
        self._raises = raises

    def available(self):
        return self._available, self._detail

    def score_batch(self, texts):
        if self._raises:
            raise self._raises
        return [self._score] * len(texts)


def _patch_detectors(monkeypatch, scores):
    """scores: {name: value_or_StubDetector}"""
    def fake_get_detector(name):
        v = scores[name]
        return v if isinstance(v, StubDetector) else StubDetector(score=v)
    monkeypatch.setattr("core.detectors.get_detector", fake_get_detector)


# --- artifact loading --------------------------------------------------------

def test_policy_unavailable_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fusion_mod, "ARTIFACT_FILE", str(tmp_path / "nope.json"))
    ok, detail = fusion_mod.policy_available()
    assert ok is False
    assert "not found" in detail


def test_policy_loads_successfully(artifact_file):
    ok, detail = fusion_mod.policy_available()
    assert ok is True
    assert "4 features" in detail


def test_corrupt_artifact_is_unavailable_not_fatal(tmp_path, monkeypatch):
    path = tmp_path / "fusion_policy.json"
    path.write_text("{ not valid json", encoding="utf-8")
    monkeypatch.setattr(fusion_mod, "ARTIFACT_FILE", str(path))
    ok, detail = fusion_mod.policy_available()
    assert ok is False


def test_artifact_missing_required_field_is_rejected(tmp_path, monkeypatch):
    bad = dict(ARTIFACT)
    del bad["threshold_high"]
    path = tmp_path / "fusion_policy.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(fusion_mod, "ARTIFACT_FILE", str(path))
    ok, detail = fusion_mod.policy_available()
    assert ok is False
    assert "threshold_high" in detail


def test_mismatched_feature_lengths_rejected(tmp_path, monkeypatch):
    bad = dict(ARTIFACT)
    bad["coefficients"] = [1.0, 2.0]  # only 2, but 4 features declared
    path = tmp_path / "fusion_policy.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(fusion_mod, "ARTIFACT_FILE", str(path))
    ok, detail = fusion_mod.policy_available()
    assert ok is False
    assert "mismatch" in detail


def test_policy_loaded_only_once(artifact_file):
    fusion_mod.policy_available()
    first_obj = fusion_mod._policy
    artifact_file.write_text(json.dumps({**ARTIFACT, "threshold_high": 0.99}), encoding="utf-8")
    fusion_mod.policy_available()
    assert fusion_mod._policy is first_obj  # not reloaded


# --- scoring: the happy path --------------------------------------------------

def test_fused_score_combines_all_four_features(artifact_file, monkeypatch):
    _patch_detectors(monkeypatch, {
        "protectai_injection": 0.9, "madhurjindal_jailbreak": 0.8, "toxic_bert": 0.1,
    })
    result = fusion_mod.fused_threat_score("some attack text", anchor_score=0.7)

    assert result["available"] is True
    assert 0.0 <= result["score"] <= 1.0
    assert result["threshold_high"] == 0.28
    assert result["threshold_medium"] == 0.10
    assert result["detector_scores"] == {
        "anchors": 0.7, "protectai_injection": 0.9,
        "madhurjindal_jailbreak": 0.8, "toxic_bert": 0.1,
    }


def test_manual_sigmoid_matches_apply_policy(artifact_file):
    """Pins the exact linear-algebra contract against a hand-computed value."""
    fusion_mod.policy_available()
    values = {"anchors": 0.7, "protectai_injection": 0.9,
              "madhurjindal_jailbreak": 0.8, "toxic_bert": 0.1}

    z = ARTIFACT["intercept"]
    for name, mean, scale, coef in zip(
        ARTIFACT["feature_order"], ARTIFACT["scaler_mean"],
        ARTIFACT["scaler_scale"], ARTIFACT["coefficients"],
    ):
        z += coef * ((values[name] - mean) / scale)
    import math
    expected = 1.0 / (1.0 + math.exp(-z))

    assert fusion_mod._apply_policy(values) == pytest.approx(expected, abs=1e-9)


def test_higher_detector_scores_yield_higher_fused_score(artifact_file, monkeypatch):
    """Sanity: the policy must be monotonic in the direction attacks push it."""
    _patch_detectors(monkeypatch, {
        "protectai_injection": 0.0, "madhurjindal_jailbreak": 0.0, "toxic_bert": 0.0,
    })
    low = fusion_mod.fused_threat_score("benign", anchor_score=0.0)["score"]

    fusion_mod._policy_loaded = False  # allow re-probing detectors for 2nd call
    _patch_detectors(monkeypatch, {
        "protectai_injection": 1.0, "madhurjindal_jailbreak": 1.0, "toxic_bert": 1.0,
    })
    high = fusion_mod.fused_threat_score("attack", anchor_score=1.0)["score"]

    assert high > low


# --- fail-closed-to-fallback: the critical contract --------------------------

def test_unavailable_detector_reports_unavailable_not_a_score(artifact_file, monkeypatch):
    _patch_detectors(monkeypatch, {
        "protectai_injection": StubDetector(available=False, detail="gated repo"),
        "madhurjindal_jailbreak": 0.5, "toxic_bert": 0.1,
    })
    result = fusion_mod.fused_threat_score("text", anchor_score=0.5)

    assert result["available"] is False
    assert result["score"] is None
    assert "protectai_injection" in result["detail"]
    assert "gated repo" in result["detail"]


def test_missing_detector_never_silently_becomes_zero(artifact_file, monkeypatch):
    """
    THE CONTRACT THAT MATTERS MOST. An unavailable detector must not be
    imputed as 0.0 — that would understate risk exactly when a detector goes
    dark, which is the wrong direction to fail for a security control.
    """
    _patch_detectors(monkeypatch, {
        "protectai_injection": StubDetector(available=False, detail="oom"),
        "madhurjindal_jailbreak": 0.9, "toxic_bert": 0.9,
    })
    result = fusion_mod.fused_threat_score("dangerous text", anchor_score=0.9)

    assert result["available"] is False
    assert result["score"] is None  # NOT 0.0, NOT a partial computation


def test_detector_exception_falls_back_cleanly(artifact_file, monkeypatch):
    _patch_detectors(monkeypatch, {
        "protectai_injection": StubDetector(raises=RuntimeError("cuda oom")),
        "madhurjindal_jailbreak": 0.5, "toxic_bert": 0.1,
    })
    result = fusion_mod.fused_threat_score("text", anchor_score=0.5)

    assert result["available"] is False
    assert "RuntimeError" in result["detail"]


def test_missing_artifact_short_circuits_before_touching_any_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(fusion_mod, "ARTIFACT_FILE", str(tmp_path / "nope.json"))

    def boom(name):
        raise AssertionError("must not probe detectors when the policy itself is unavailable")
    monkeypatch.setattr("core.detectors.get_detector", boom)

    result = fusion_mod.fused_threat_score("text", anchor_score=0.5)
    assert result["available"] is False
    assert result["detector_scores"] == {}


def test_partial_detector_scores_are_still_reported_on_failure(artifact_file, monkeypatch):
    """
    Even when unavailable, whatever WAS computed before the failure should be
    visible for debugging — but must never be mistaken for a usable score.
    """
    _patch_detectors(monkeypatch, {
        "protectai_injection": 0.4,
        "madhurjindal_jailbreak": StubDetector(available=False, detail="down"),
        "toxic_bert": 0.1,
    })
    result = fusion_mod.fused_threat_score("text", anchor_score=0.6)

    assert result["available"] is False
    assert result["detector_scores"]["anchors"] == 0.6
    assert result["detector_scores"]["protectai_injection"] == 0.4
    assert "madhurjindal_jailbreak" not in result["detector_scores"]
