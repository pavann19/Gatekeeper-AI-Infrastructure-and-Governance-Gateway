"""
Tests for EmbeddingHeadDetector (issue #4) — the wrapper around this
project's own multilingual_head feature (embedding + logistic-regression
head fitted on data/eval_suite.jsonl, not a pretrained public classifier).

The encoder is mocked throughout, same discipline as test_detectors.py's
TransformerDetector tests: these verify the wiring (artifact loading,
manual logistic-regression application, availability contract), not the
real model's classification quality — that is what scripts.build_
multilingual_feature's held-out/leave-one-source-out numbers and
scripts.validate_multilingual_head_nested's disjoint-split numbers are for.

CI DOES NOT INSTALL sentence-transformers (see requirements-ci.txt: "no
heavy ML MODEL DOWNLOADS needed for unit tests — sentence-transformers
and real HF weights are still excluded/mocked"), so these tests must
never `import sentence_transformers` for real — `_install_fake_encoder`
injects a fake module into `sys.modules` instead, which satisfies
`core/detectors.py`'s local `from sentence_transformers import
SentenceTransformer` without the real package ever needing to exist.
"""
import json
import math
import sys
import types

import numpy as np
import pytest

from core.detectors import EmbeddingHeadDetector


def _write_artifact(path, dim=4, coefficients=None, intercept=0.0,
                    scaler_mean=None, scaler_scale=None):
    coefficients = coefficients if coefficients is not None else [1.0] * dim
    scaler_mean = scaler_mean if scaler_mean is not None else [0.0] * dim
    scaler_scale = scaler_scale if scaler_scale is not None else [1.0] * dim
    artifact = {
        "version": 1,
        "model_id": "fake/multilingual-encoder",
        "embedding_dim": dim,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "coefficients": coefficients,
        "intercept": intercept,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer: maps each text to a
    fixed vector via its length, so tests can compute the expected score
    by hand without downloading a real model."""

    def __init__(self, vectors_by_text=None, dim=4):
        self.vectors_by_text = vectors_by_text or {}
        self.dim = dim

    def encode(self, texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True):
        return np.array([self.vectors_by_text.get(t, [0.0] * self.dim) for t in texts])


def _install_fake_encoder(monkeypatch, encoder):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = lambda model_id, **kw: encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


# --- artifact loading --------------------------------------------------------

def test_unavailable_when_artifact_missing(tmp_path):
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(tmp_path / "nope.json"),
                              targets=("prompt_injection",))
    ok, detail = d.available()
    assert ok is False
    assert "not found" in detail


def test_corrupt_artifact_is_unavailable_not_fatal(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{ not valid json", encoding="utf-8")
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    ok, detail = d.available()
    assert ok is False


def test_artifact_missing_required_field_is_rejected(tmp_path):
    path = tmp_path / "artifact.json"
    bad = {"model_id": "x", "scaler_mean": [0.0], "scaler_scale": [1.0]}  # no coefficients/intercept
    path.write_text(json.dumps(bad), encoding="utf-8")
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    ok, detail = d.available()
    assert ok is False
    assert "coefficients" in detail or "intercept" in detail


def test_mismatched_lengths_rejected(tmp_path):
    path = tmp_path / "artifact.json"
    _write_artifact(path, dim=4, coefficients=[1.0, 2.0])  # only 2, but dim=4 scaler arrays
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    ok, detail = d.available()
    assert ok is False
    assert "mismatch" in detail


def test_available_when_artifact_and_encoder_load_cleanly(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    _write_artifact(path)
    _install_fake_encoder(monkeypatch, FakeEncoder())
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    ok, detail = d.available()
    assert ok is True
    assert "fake/multilingual-encoder" in detail


def test_trained_on_is_always_empty():
    """See EmbeddingHeadDetector's docstring: `trained_on` doesn't apply
    the normal way here (the artifact is refit on 100% of the suite), so
    it must always be the empty tuple, never a partial exclusion list."""
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path="irrelevant",
                              targets=("prompt_injection",))
    assert d.trained_on == ()


# --- scoring: manual logistic regression matches hand computation -----------

def test_score_matches_hand_computed_sigmoid(tmp_path, monkeypatch):
    coefficients = [0.5, -1.0, 2.0, 0.1]
    scaler_mean = [0.1, 0.2, 0.3, 0.4]
    scaler_scale = [1.0, 2.0, 0.5, 4.0]
    intercept = -0.3
    path = tmp_path / "artifact.json"
    _write_artifact(path, coefficients=coefficients, intercept=intercept,
                    scaler_mean=scaler_mean, scaler_scale=scaler_scale)

    vec = [0.9, -0.4, 0.7, 1.2]
    _install_fake_encoder(monkeypatch, FakeEncoder(vectors_by_text={"attack text": vec}))

    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    [score] = d.score_batch(["attack text"])

    z = intercept
    for x, mean, scale, coef in zip(vec, scaler_mean, scaler_scale, coefficients):
        z += coef * ((x - mean) / scale)
    expected = 1.0 / (1.0 + math.exp(-z))

    assert score == pytest.approx(expected, abs=1e-9)


def test_score_batch_preserves_order(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    _write_artifact(path)
    encoder = FakeEncoder(vectors_by_text={
        "low": [0.0, 0.0, 0.0, 0.0],
        "high": [5.0, 5.0, 5.0, 5.0],
    })
    _install_fake_encoder(monkeypatch, encoder)
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    low_score, high_score = d.score_batch(["low", "high"])
    assert high_score > low_score


def test_scores_are_bounded_probabilities(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    _write_artifact(path, coefficients=[100.0] * 4)  # deliberately extreme
    _install_fake_encoder(monkeypatch, FakeEncoder(vectors_by_text={"x": [10.0] * 4}))
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    [score] = d.score_batch(["x"])
    assert 0.0 <= score <= 1.0


def test_blank_text_does_not_crash(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    _write_artifact(path)
    _install_fake_encoder(monkeypatch, FakeEncoder())
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(path),
                              targets=("prompt_injection",))
    scores = d.score_batch([""])
    assert len(scores) == 1


def test_score_batch_raises_when_artifact_missing(tmp_path):
    d = EmbeddingHeadDetector(name="multilingual_head", artifact_path=str(tmp_path / "nope.json"),
                              targets=("prompt_injection",))
    with pytest.raises(RuntimeError, match="unavailable"):
        d.score_batch(["text"])


# --- registry integration ----------------------------------------------------

def test_registered_in_the_default_registry():
    from core.detectors import get_registry
    reg = get_registry()
    assert "multilingual_head" in reg
    assert isinstance(reg["multilingual_head"], EmbeddingHeadDetector)
    assert reg["multilingual_head"].trained_on == ()
    assert reg["multilingual_head"].targets
    assert reg["multilingual_head"].description
