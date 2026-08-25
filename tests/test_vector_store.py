"""
Tests for core/vector_store.py's FAISS-backed ScalableVectorStore.
Uses the real faiss library with deterministic fake embeddings (monkeypatched
get_embedding) so similarity/ordering assertions reflect real cosine-similarity
math, not mocked-out behavior.
"""
import numpy as np
import torch
import pytest

from core.vector_store import ScalableVectorStore
import core.vector_store as vector_store_mod


def _vec(*components, dim=4):
    """Builds a torch tensor of length `dim`, zero-padded, mimicking get_embedding's output."""
    arr = np.zeros(dim, dtype=np.float32)
    for i, c in enumerate(components):
        arr[i] = c
    return torch.tensor(arr)


def _fake_embedder(mapping):
    """Returns a function usable as get_embedding, looking up text in mapping."""
    def _embed(text):
        return mapping[text]
    return _embed


def test_empty_store_similarity_is_zero():
    store = ScalableVectorStore(dimension=4)
    query = _vec(1, 0, 0, 0)
    assert store.get_max_similarity(query) == 0.0


def test_empty_store_centroid_is_none():
    store = ScalableVectorStore(dimension=4)
    assert store.get_centroid() is None


def test_add_texts_with_empty_list_is_noop():
    store = ScalableVectorStore(dimension=4)
    store.add_texts([])
    assert store.texts == []
    assert store.get_max_similarity(_vec(1, 0, 0, 0)) == 0.0


def test_max_similarity_exact_match_is_one(monkeypatch):
    mapping = {"cat": _vec(1, 0, 0, 0)}
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=4)
    store.add_texts(["cat"])

    score = store.get_max_similarity(_vec(1, 0, 0, 0))
    assert score == pytest.approx(1.0, abs=1e-5)


def test_max_similarity_orthogonal_vector_is_zero(monkeypatch):
    mapping = {"cat": _vec(1, 0, 0, 0)}
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=4)
    store.add_texts(["cat"])

    score = store.get_max_similarity(_vec(0, 1, 0, 0))
    assert score == pytest.approx(0.0, abs=1e-5)


def test_max_similarity_picks_nearest_of_multiple_known_vectors(monkeypatch):
    # Three anchors at known angles from the x-axis query vector.
    mapping = {
        "close": _vec(1, 0.05, 0, 0),   # nearly identical -> highest similarity
        "far": _vec(0, 1, 0, 0),        # orthogonal -> similarity ~0
        "opposite": _vec(-1, 0, 0, 0),  # opposite -> similarity -1
    }
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=4)
    store.add_texts(["far", "opposite", "close"])

    query = _vec(1, 0, 0, 0)
    score = store.get_max_similarity(query)

    # Independently compute expected best cosine similarity (normalized "close" vs query).
    close_vec = mapping["close"].numpy()
    close_norm = close_vec / np.linalg.norm(close_vec)
    query_norm = query.numpy() / np.linalg.norm(query.numpy())
    expected = float(np.dot(close_norm, query_norm))

    assert score == pytest.approx(expected, abs=1e-4)
    assert score > 0.9  # confirms "close" (not "far" or "opposite") won


def test_add_texts_accumulates_across_multiple_calls(monkeypatch):
    mapping = {
        "a": _vec(1, 0, 0, 0),
        "b": _vec(0, 1, 0, 0),
    }
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=4)
    store.add_texts(["a"])
    store.add_texts(["b"])

    assert store.texts == ["a", "b"]
    assert store._get_index().ntotal == 2

    # Query matching "b" exactly should now find it as the max, not "a".
    score = store.get_max_similarity(_vec(0, 1, 0, 0))
    assert score == pytest.approx(1.0, abs=1e-5)


def test_index_is_lazily_created():
    store = ScalableVectorStore(dimension=4)
    assert store._index is None
    store._get_index()
    assert store._index is not None
    assert store._index.ntotal == 0


def test_get_centroid_averages_and_normalizes(monkeypatch):
    # Two orthogonal unit vectors -> mean is (0.5, 0.5, 0, 0), normalized to unit length.
    mapping = {
        "a": _vec(1, 0, 0, 0),
        "b": _vec(0, 1, 0, 0),
    }
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=4)
    store.add_texts(["a", "b"])

    centroid = store.get_centroid()
    assert centroid is not None
    assert centroid.shape == (4,)

    # Should be unit-normalized.
    assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)
    # Direction should match normalized (1,1,0,0).
    expected_dir = np.array([1, 1, 0, 0], dtype=np.float32)
    expected_dir /= np.linalg.norm(expected_dir)
    np.testing.assert_allclose(centroid, expected_dir, atol=1e-5)


def test_get_centroid_single_text_matches_that_vector_normalized(monkeypatch):
    mapping = {"only": _vec(3, 4, 0, 0)}  # norm 5
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=4)
    store.add_texts(["only"])

    centroid = store.get_centroid()
    expected = np.array([3, 4, 0, 0], dtype=np.float32) / 5.0
    np.testing.assert_allclose(centroid, expected, atol=1e-5)


def test_dimension_defaults_to_768():
    store = ScalableVectorStore()
    assert store.dimension == 768


def test_custom_dimension_is_respected(monkeypatch):
    mapping = {"x": _vec(1, 0, dim=16)}
    monkeypatch.setattr(vector_store_mod, "get_embedding", _fake_embedder(mapping))
    store = ScalableVectorStore(dimension=16)
    store.add_texts(["x"])
    assert store._get_index().d == 16
