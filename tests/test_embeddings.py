"""
Tests for core/embeddings.py: lazy-singleton model loading, embedding
generation, and cosine similarity.

Uses a lightweight fake model (monkeypatched into the module-level cache)
for the singleton/caching tests, since a real SentenceTransformer load is
slow. A small set of integration tests load the real cached
all-mpnet-base-v2 model (already present in this machine's HF cache) to
verify actual output shape/behaviour.
"""
import pytest
import torch

import core.embeddings as embeddings_mod
from core.embeddings import cosine_similarity, get_embedding

# core/embeddings.py's cosine_similarity() unconditionally imports the real
# `sentence_transformers.util` (and _get_model() the real SentenceTransformer
# class), so any test exercising either -- even with fake vectors, not just
# the "real model" integration tests below -- needs the real package
# installed. CI's requirements-ci.txt deliberately excludes it (no network/
# GB-scale weights budget in CI); these tests skip cleanly there rather than
# failing, matching this project's own precedent in test_llama_guard.py's
# `torch = pytest.importorskip("torch")`.
pytest.importorskip("sentence_transformers")


def _reset_model_cache(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_model", None)


class _FakeModel:
    """Minimal stand-in for SentenceTransformer: deterministic, no download."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, text, convert_to_tensor=False):
        self.encode_calls.append(text)
        # Deterministic vector derived from text length/hash so equal inputs
        # produce equal vectors and different inputs (usually) differ.
        seed = float(len(text) % 7 + 1)
        vec = torch.tensor([seed, seed * 2, seed * 3])
        return vec if convert_to_tensor else vec.tolist()


# --- _get_model: lazy singleton --------------------------------------------

def test_model_not_loaded_until_first_use(monkeypatch):
    _reset_model_cache(monkeypatch)
    assert embeddings_mod._model is None


def test_get_model_returns_same_cached_instance(monkeypatch):
    _reset_model_cache(monkeypatch)
    fake = _FakeModel()
    load_calls = {"n": 0}

    def fake_constructor(name):
        load_calls["n"] += 1
        return fake

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", fake_constructor
    )

    first = embeddings_mod._get_model()
    second = embeddings_mod._get_model()

    assert first is fake
    assert second is first
    assert load_calls["n"] == 1  # constructed only once despite two calls


def test_get_model_uses_configured_embedding_model_name(monkeypatch):
    _reset_model_cache(monkeypatch)
    seen_names = []

    def fake_constructor(name):
        seen_names.append(name)
        return _FakeModel()

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", fake_constructor
    )
    embeddings_mod._get_model()
    assert seen_names == [embeddings_mod.EMBEDDING_MODEL]


def test_model_load_failure_propagates_and_does_not_cache(monkeypatch):
    _reset_model_cache(monkeypatch)

    def failing_constructor(name):
        raise OSError("model not found")

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", failing_constructor
    )

    try:
        embeddings_mod._get_model()
        assert False, "expected OSError to propagate"
    except OSError:
        pass

    # A failed load must not leave a broken model cached -- the next call
    # should retry construction rather than silently return None or a
    # half-built object.
    assert embeddings_mod._model is None


# --- get_embedding -----------------------------------------------------------

def test_get_embedding_calls_encode_on_cached_model(monkeypatch):
    _reset_model_cache(monkeypatch)
    fake = _FakeModel()
    monkeypatch.setattr(embeddings_mod, "_get_model", lambda: fake)

    result = get_embedding("hello world")

    assert fake.encode_calls == ["hello world"]
    assert isinstance(result, torch.Tensor)


def test_get_embedding_is_deterministic_for_same_input(monkeypatch):
    _reset_model_cache(monkeypatch)
    fake = _FakeModel()
    monkeypatch.setattr(embeddings_mod, "_get_model", lambda: fake)

    v1 = get_embedding("repeat me")
    v2 = get_embedding("repeat me")
    assert torch.equal(v1, v2)


def test_get_embedding_differs_for_different_inputs(monkeypatch):
    _reset_model_cache(monkeypatch)
    fake = _FakeModel()
    monkeypatch.setattr(embeddings_mod, "_get_model", lambda: fake)

    v1 = get_embedding("short")
    v2 = get_embedding("a much longer piece of text than the other one")
    assert not torch.equal(v1, v2)


def test_get_embedding_handles_empty_string(monkeypatch):
    _reset_model_cache(monkeypatch)
    fake = _FakeModel()
    monkeypatch.setattr(embeddings_mod, "_get_model", lambda: fake)

    result = get_embedding("")
    assert fake.encode_calls == [""]
    assert isinstance(result, torch.Tensor)


def test_get_embedding_handles_very_long_input(monkeypatch):
    _reset_model_cache(monkeypatch)
    fake = _FakeModel()
    monkeypatch.setattr(embeddings_mod, "_get_model", lambda: fake)

    long_text = "word " * 5000
    result = get_embedding(long_text)
    assert fake.encode_calls == [long_text]
    assert isinstance(result, torch.Tensor)


# --- cosine_similarity --------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    v = torch.tensor([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    assert abs(cosine_similarity(v1, v2)) < 1e-6


def test_cosine_similarity_opposite_vectors_is_negative_one():
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([-1.0, -2.0, -3.0])
    assert abs(cosine_similarity(v1, v2) - (-1.0)) < 1e-6


def test_cosine_similarity_accepts_plain_lists_not_just_tensors():
    # get_embedding(..., convert_to_tensor=False) callers, or raw list
    # inputs from any other caller, must still work per the isinstance
    # checks in cosine_similarity.
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-6


def test_cosine_similarity_mixed_list_and_tensor_inputs():
    v1 = [1.0, 1.0]
    v2 = torch.tensor([1.0, 1.0])
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-6


def test_cosine_similarity_returns_python_float():
    v1 = torch.tensor([1.0, 2.0])
    v2 = torch.tensor([2.0, 1.0])
    result = cosine_similarity(v1, v2)
    assert isinstance(result, float)


# --- Integration: real cached all-mpnet-base-v2 model -------------------------
# EMBEDDING_MODEL's default is already present in this machine's HF cache
# (no network download needed), so these exercise the real encode() path
# rather than only the fake stand-in above.

def test_real_model_produces_768_dim_vector_for_mpnet(monkeypatch):
    _reset_model_cache(monkeypatch)
    vec = get_embedding("The quick brown fox jumps over the lazy dog.")
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (768,)  # all-mpnet-base-v2's known output dimensionality


def test_real_model_is_deterministic_across_calls(monkeypatch):
    _reset_model_cache(monkeypatch)
    v1 = get_embedding("Deterministic embedding check.")
    v2 = get_embedding("Deterministic embedding check.")
    assert torch.equal(v1, v2)


def test_real_model_semantic_similarity_ranks_related_text_higher(monkeypatch):
    _reset_model_cache(monkeypatch)
    anchor = get_embedding("The cat sat on the mat.")
    related = get_embedding("A kitten was resting on the rug.")
    unrelated = get_embedding("Quarterly tax filings are due in April.")

    sim_related = cosine_similarity(anchor, related)
    sim_unrelated = cosine_similarity(anchor, unrelated)
    assert sim_related > sim_unrelated


def test_real_model_handles_empty_string_without_error(monkeypatch):
    _reset_model_cache(monkeypatch)
    vec = get_embedding("")
    assert vec.shape == (768,)


def test_real_model_second_get_model_call_reuses_cached_instance(monkeypatch):
    _reset_model_cache(monkeypatch)
    m1 = embeddings_mod._get_model()
    m2 = embeddings_mod._get_model()
    assert m1 is m2
