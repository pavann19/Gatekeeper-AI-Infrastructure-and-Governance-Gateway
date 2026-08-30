"""
Tests for detector model revision pinning, hash integrity verification, and auth error logging.
"""
from __future__ import annotations

import json
import logging
import os
import unittest.mock as mock
import pytest

from core.config import settings
from core.detectors import (
    AnchorDetector,
    EmbeddingHeadDetector,
    LlamaGuardDetector,
    ModelIntegrityError,
    NemoGuardJailbreakDetector,
    TransformerDetector,
    _is_auth_error,
    _verify_detector_integrity,
    get_registry,
)


# ---------------------------------------------------------------------------
# (a) Every registry entry has a pinned revision
# ---------------------------------------------------------------------------

def test_every_model_detector_in_registry_has_pinned_revision():
    """All model-backed detectors in the registry must have an explicit pinned
    revision (commit SHA or tag, not a floating branch like 'main').
    """
    registry = get_registry()
    floating_branches = {"main", "master", "dev", "latest", "head", ""}

    for name, detector in registry.items():
        if isinstance(detector, AnchorDetector):
            continue  # Baseline anchor detector uses local FAISS / sentence embeddings

        assert hasattr(detector, "revision"), f"Detector '{name}' has no .revision attribute"
        revision = detector.revision
        assert isinstance(revision, str), f"Detector '{name}' revision is not a string: {revision!r}"
        assert len(revision.strip()) >= 7, f"Detector '{name}' revision '{revision}' is too short"
        assert revision.lower() not in floating_branches, (
            f"Detector '{name}' uses floating branch '{revision}' instead of pinned commit SHA"
        )


def test_detector_manifest_covers_all_model_detectors():
    """models/detector_manifest.json must exist and declare the expected revision
    and SHA256 for each model-backed detector in the registry.
    """
    manifest_path = os.path.join("models", "detector_manifest.json")
    assert os.path.exists(manifest_path), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    registry = get_registry()
    for name, detector in registry.items():
        if isinstance(detector, AnchorDetector):
            continue
        assert name in manifest, f"Detector '{name}' missing from {manifest_path}"
        entry = manifest[name]
        assert "revision" in entry, f"Manifest entry '{name}' missing 'revision'"
        assert entry["revision"] == detector.revision, (
            f"Manifest revision mismatch for '{name}': {entry['revision']} vs {detector.revision}"
        )


# ---------------------------------------------------------------------------
# (b) With DETECTOR_VERIFY_HASHES=True, mismatch raises loud error
# ---------------------------------------------------------------------------

def test_verify_hashes_loud_error_on_sha256_mismatch(tmp_path, monkeypatch):
    """When DETECTOR_VERIFY_HASHES is True and a model's weight hash does not match
    the manifest, loading must raise a distinct, loud ModelIntegrityError — not a
    warning and not a silent available=False.
    """
    monkeypatch.setattr(settings, "DETECTOR_VERIFY_HASHES", True)

    # Create a corrupted manifest with an invalid sha256
    bad_manifest = {
        "protectai_injection": {
            "model_id": "protectai/deberta-v3-base-prompt-injection-v2",
            "revision": "90c9989b1a342275dd0d1a95aad283c04e075671",
            "filename": "model.safetensors",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        }
    }
    bad_manifest_file = str(tmp_path / "bad_manifest.json")
    with open(bad_manifest_file, "w", encoding="utf-8") as f:
        json.dump(bad_manifest, f)

    # Calling _verify_detector_integrity with bad manifest must raise ModelIntegrityError
    with pytest.raises(ModelIntegrityError, match="integrity verification failed|SHA256 mismatch"):
        _verify_detector_integrity(
            name="protectai_injection",
            model_id="protectai/deberta-v3-base-prompt-injection-v2",
            revision="90c9989b1a342275dd0d1a95aad283c04e075671",
            manifest_path=bad_manifest_file,
        )

    # Also test that TransformerDetector._load() re-raises ModelIntegrityError loudly
    detector = TransformerDetector(
        name="protectai_injection",
        model_id="protectai/deberta-v3-base-prompt-injection-v2",
        revision="90c9989b1a342275dd0d1a95aad283c04e075671",
        positive_labels=["injection"],
        targets=("prompt_injection",),
    )

    # Mock _verify_detector_integrity to raise ModelIntegrityError
    monkeypatch.setattr(
        "core.detectors._verify_detector_integrity",
        mock.MagicMock(side_effect=ModelIntegrityError("Tampered weight file detected")),
    )

    with pytest.raises(ModelIntegrityError, match="Tampered weight file detected"):
        detector._load()


def test_verify_hashes_loud_error_on_revision_mismatch(tmp_path):
    """When a model's revision differs from the manifest, a loud ModelIntegrityError
    must be raised immediately.
    """
    manifest_data = {
        "test_model": {
            "model_id": "test/model",
            "revision": "expected_commit_hash_12345",
            "sha256": "abcdef",
        }
    }
    manifest_file = str(tmp_path / "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    with pytest.raises(ModelIntegrityError, match="revision mismatch"):
        _verify_detector_integrity(
            name="test_model",
            model_id="test/model",
            revision="wrong_commit_hash_67890",
            manifest_path=manifest_file,
        )


# ---------------------------------------------------------------------------
# (c) With correct manifest, it loads clean
# ---------------------------------------------------------------------------

def test_verify_hashes_clean_on_correct_manifest(monkeypatch):
    """When DETECTOR_VERIFY_HASHES is True and the manifest matches, verification
    completes cleanly with no exceptions.
    """
    monkeypatch.setattr(settings, "DETECTOR_VERIFY_HASHES", True)

    # Test verification against cached models
    _verify_detector_integrity(
        name="protectai_injection",
        model_id="protectai/deberta-v3-base-prompt-injection-v2",
        revision="90c9989b1a342275dd0d1a95aad283c04e075671",
    )
    _verify_detector_integrity(
        name="toxic_bert",
        model_id="unitary/toxic-bert",
        revision="4d6c22e74ba2fdd26bc4f7238f50766b045a0d94",
    )


# ---------------------------------------------------------------------------
# (d) Gated / auth error logging at ERROR level vs info on success
# ---------------------------------------------------------------------------

def test_auth_error_classifier_helper():
    """_is_auth_error accurately identifies HuggingFace 401/403/Gated exceptions."""
    class MockGatedRepoError(Exception):
        pass

    assert _is_auth_error(MockGatedRepoError("Cannot access gated repo for model")) is True
    assert _is_auth_error(Exception("401 Client Error: Unauthorized for url")) is True
    assert _is_auth_error(Exception("403 Client Error: Forbidden for url")) is True
    assert _is_auth_error(Exception("RepositoryNotFound: Access to model is restricted")) is True
    assert _is_auth_error(FileNotFoundError("No such file or directory")) is False
    assert _is_auth_error(ValueError("scaler/coefficients length mismatch")) is False


def test_gated_model_auth_failure_logs_at_error_level(caplog):
    """When a gated model fails to load due to auth (e.g. 401 or GatedRepoError),
    it must log at ERROR level, clearly distinct from warnings or successful load logs.
    """
    detector = TransformerDetector(
        name="prompt_guard_2",
        model_id="meta-llama/Llama-Prompt-Guard-2-86M",
        revision="a8ded8e697ce7c355e395a0df51f94adb4a2fd27",
        positive_labels=["injection", "jailbreak"],
        targets=("prompt_injection", "jailbreak"),
    )

    with mock.patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=Exception("401 Client Error: Cannot access gated repo meta-llama/Llama-Prompt-Guard-2-86M"),
    ):
        with caplog.at_level(logging.DEBUG):
            detector._load()

        assert detector._model is None
        assert "401" in detector._load_error

        # Assert an ERROR log was emitted with auth/gated context
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert any("unavailable due to authentication/licence" in r.message for r in error_records)
        assert any("prompt_guard_2" in r.message for r in error_records)


def test_successful_load_logs_verified_and_loaded(caplog):
    """When a detector loads successfully, it logs 'verified and loaded' at INFO level."""
    detector = TransformerDetector(
        name="test_det",
        model_id="fake/test-model",
        revision="abcdef123456",
        positive_labels=["injection"],
        targets=("prompt_injection",),
    )

    mock_tokenizer = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_model.config.id2label = {0: "safe", 1: "injection"}

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         mock.patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
        with caplog.at_level(logging.INFO):
            detector._load()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("verified and loaded" in r.message for r in info_records)
        assert any("test_det" in r.message for r in info_records)
