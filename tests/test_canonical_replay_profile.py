"""Negative controls for canonical replay: no silent degraded-model acceptance."""
import json
import pytest

from scripts import canonical_replay as canonical


def test_artifact_drift_is_rejected(tmp_path):
    artifact = tmp_path / "policy.json"
    artifact.write_text("{}")
    profile = {"artifacts": {str(artifact): "not-the-hash"}}
    with pytest.raises(canonical.CanonicalEnvironmentError, match="artifact"):
        canonical.validate_artifacts(profile)


def test_optional_detector_cannot_silently_change_canonical_tier():
    class Missing:
        revision = "pinned"
        def available(self):
            return False, "missing asset"

    profile = {"detectors": {"required": {"revision": "pinned"}}}
    with pytest.raises(canonical.CanonicalEnvironmentError, match="required"):
        canonical.validate_detectors(profile, {"required": Missing()}, verify_weights=False)


def test_revision_substitution_is_rejected():
    class Wrong:
        revision = "floating"
        def available(self):
            return True, "loaded"

    profile = {"detectors": {"required": {"revision": "pinned"}}}
    with pytest.raises(canonical.CanonicalEnvironmentError, match="revision"):
        canonical.validate_detectors(profile, {"required": Wrong()}, verify_weights=False)


def test_configuration_drift_is_rejected():
    class Config:
        META_INTENT_THRESHOLD = 0.1

    with pytest.raises(canonical.CanonicalEnvironmentError, match="META_INTENT_THRESHOLD"):
        canonical.validate_configuration({"settings": {"META_INTENT_THRESHOLD": 0.3}}, Config())


def test_profile_frozen_oracle_is_not_regenerated():
    profile = canonical.load_profile()
    canonical.validate_artifacts(profile)
    assert profile["fusion_tier"] == "nine_feature"
    assert profile["judge_state"] == "unavailable"
    assert profile["cache_state"] == "bypassed"
    assert len(profile["detectors"]) == 8
    assert profile["seed"] == 42
    assert json.loads(open("models/fusion_policy.json", encoding="utf-8").read())["version"] == 3


@pytest.mark.parametrize("details", [
    {"fusion_available": False},
    {"fusion_available": True, "fusion_detail": "fusion applied, tier=six_feature"},
])
def test_mid_replay_detector_failure_cannot_become_canonical_fallback(details):
    with pytest.raises(canonical.CanonicalEnvironmentError):
        canonical.validate_decision_tier({"fusion_tier": "nine_feature"}, details, "case")


def test_symbolic_and_fast_path_do_not_require_per_row_fusion_scores():
    canonical.validate_decision_tier({"fusion_tier": "nine_feature"}, {}, "case")


def test_public_replay_pins_consumed_inputs_without_private_training_dependency():
    profile = canonical.load_profile()
    assert "data/eval_suite.jsonl" not in profile["artifacts"]
    assert "tests/fixtures/decision_replay_corpus.jsonl" in profile["artifacts"]
    assert profile["training_provenance"]["artifacts"] == {
        "data/eval_suite.jsonl": "d6532f645982531a4ffa13ba7ec0bb4d13cce32dc458b9e4efad55c2ece7ef1d"
    }
    canonical.validate_artifacts(profile)


def test_explicit_training_provenance_rejects_missing_or_corrupt_input(tmp_path, monkeypatch):
    provenance = canonical.load_profile()["training_provenance"]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(canonical.CanonicalEnvironmentError, match="data/eval_suite.jsonl"):
        canonical.validate_artifacts(provenance)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/eval_suite.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(canonical.CanonicalEnvironmentError, match="data/eval_suite.jsonl"):
        canonical.validate_artifacts(provenance)
