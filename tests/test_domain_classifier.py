"""
Tests for core/domain_classifier.py — the topicality (domain alignment)
signal used by core/risk.py's fuse_signals() and surfaced via api's
`topicality` field. NOT a safety signal (see docs/EVALUATION_METHODOLOGY.md).

These tests mock core.embeddings.get_embedding/cosine_similarity so they
run fast and deterministically without loading a real sentence-transformers
model, while exercising the classifier's real corpus-loading, lazy-centroid,
caching, and failure-handling logic.
"""
import json
import logging

import pytest

import core.domain_classifier as dc


@pytest.fixture(autouse=True)
def reset_lazy_state(monkeypatch):
    """The module caches corpus/centroid state in globals computed on first
    use. Reset before every test so tests don't leak state into each other,
    and point DOMAIN_CORPUS_FILE at a path each test controls."""
    monkeypatch.setattr(dc, "_corpus_documents", None)
    monkeypatch.setattr(dc, "_domain_centroid_cache", None)
    monkeypatch.setattr(dc, "_centroid_initialized", False)
    yield


def _write_corpus(path, documents):
    path.write_text(json.dumps({"documents": documents}), encoding="utf-8")


# --- _load_domain_corpus -----------------------------------------------

def test_load_domain_corpus_missing_file_returns_empty_and_warns(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(missing))
    with caplog.at_level(logging.WARNING):
        result = dc._load_domain_corpus()
    assert result == []
    assert any("not found" in r.message for r in caplog.records)


def test_load_domain_corpus_reads_documents(monkeypatch, tmp_path):
    corpus_file = tmp_path / "corpus.json"
    docs = ["Explain gradient descent.", "How do firewalls filter packets?"]
    _write_corpus(corpus_file, docs)
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(corpus_file))
    result = dc._load_domain_corpus()
    assert result == docs


def test_load_domain_corpus_malformed_json_returns_empty_and_warns(monkeypatch, tmp_path, caplog):
    corpus_file = tmp_path / "corpus.json"
    corpus_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(corpus_file))
    with caplog.at_level(logging.WARNING):
        result = dc._load_domain_corpus()
    assert result == []
    assert any("Failed to load domain corpus" in r.message for r in caplog.records)


def test_load_domain_corpus_missing_documents_key_returns_empty(monkeypatch, tmp_path):
    corpus_file = tmp_path / "corpus.json"
    corpus_file.write_text(json.dumps({"other_key": []}), encoding="utf-8")
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(corpus_file))
    result = dc._load_domain_corpus()
    assert result == []


# --- _compute_centroid ---------------------------------------------------

def test_compute_centroid_none_when_no_documents():
    assert dc._compute_centroid([]) is None


def test_compute_centroid_averages_embeddings(monkeypatch):
    fake_vectors = {
        "doc a": [1.0, 0.0, 3.0],
        "doc b": [3.0, 2.0, 1.0],
    }
    monkeypatch.setattr(
        "core.embeddings.get_embedding", lambda text: fake_vectors[text]
    )
    centroid = dc._compute_centroid(["doc a", "doc b"])
    assert centroid == pytest.approx([2.0, 1.0, 2.0])


def test_compute_centroid_skips_documents_with_no_embedding(monkeypatch):
    def fake_embed(text):
        return None if text == "bad" else [1.0, 1.0]

    monkeypatch.setattr("core.embeddings.get_embedding", fake_embed)
    centroid = dc._compute_centroid(["bad", "good"])
    assert centroid == pytest.approx([1.0, 1.0])


def test_compute_centroid_all_embeddings_fail_returns_none(monkeypatch):
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: None)
    assert dc._compute_centroid(["doc a", "doc b"]) is None


def test_compute_centroid_handles_tensor_like_objects(monkeypatch):
    """Embeddings may come back as tensors with .tolist(); the module must
    convert them before doing plain arithmetic."""

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    monkeypatch.setattr(
        "core.embeddings.get_embedding", lambda text: FakeTensor([2.0, 4.0])
    )
    centroid = dc._compute_centroid(["doc a"])
    assert centroid == pytest.approx([2.0, 4.0])


# --- _get_domain_centroid (lazy load + caching) --------------------------

def test_get_domain_centroid_is_lazy_not_computed_at_import(monkeypatch, tmp_path):
    # Confirms globals truly start uninitialized (fixture already resets
    # them) and nothing is computed until the accessor is called.
    assert dc._centroid_initialized is False
    assert dc._domain_centroid_cache is None


def test_get_domain_centroid_computes_once_and_caches(monkeypatch, tmp_path):
    corpus_file = tmp_path / "corpus.json"
    _write_corpus(corpus_file, ["doc a"])
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(corpus_file))

    call_count = {"n": 0}

    def fake_embed(text):
        call_count["n"] += 1
        return [1.0, 2.0]

    monkeypatch.setattr("core.embeddings.get_embedding", fake_embed)

    first = dc._get_domain_centroid()
    second = dc._get_domain_centroid()

    assert first == pytest.approx([1.0, 2.0])
    assert second == first
    # Corpus should only be embedded once, on the first call.
    assert call_count["n"] == 1
    assert dc._centroid_initialized is True


def test_get_domain_centroid_warns_when_corpus_missing(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(missing))
    with caplog.at_level(logging.WARNING):
        centroid = dc._get_domain_centroid()
    assert centroid is None
    messages = [r.message for r in caplog.records]
    assert any("not found" in m for m in messages)
    assert any("Domain centroid not computed" in m for m in messages)


# --- is_domain_aligned: correctness, shape, bounds, edge cases ----------

def _patch_embeddings(monkeypatch, corpus_vec_map, prompt_vec_map, sim_fn=None):
    """Wire is_domain_aligned's lazy imports of core.embeddings.* to fakes."""

    def fake_get_embedding(text):
        if text in corpus_vec_map:
            return corpus_vec_map[text]
        return prompt_vec_map.get(text)

    def fake_cosine(vec1, vec2):
        if sim_fn:
            return sim_fn(vec1, vec2)
        # default: dot product normalized isn't needed for these fakes;
        # tests supply explicit sim_fn when they care about the value.
        return 1.0 if vec1 == vec2 else 0.0

    monkeypatch.setattr("core.embeddings.get_embedding", fake_get_embedding)
    monkeypatch.setattr("core.embeddings.cosine_similarity", fake_cosine)


def _setup_corpus(monkeypatch, tmp_path, docs):
    corpus_file = tmp_path / "corpus.json"
    _write_corpus(corpus_file, docs)
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(corpus_file))


def test_is_domain_aligned_in_domain_prompt_scores_above_threshold(monkeypatch, tmp_path):
    corpus_docs = ["How does a neural network backpropagate gradients?"]
    prompt = "Can you explain how gradient descent optimizes a loss function in deep learning?"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)
    _patch_embeddings(
        monkeypatch,
        corpus_vec_map={corpus_docs[0]: [1.0, 0.0]},
        prompt_vec_map={prompt: [1.0, 0.0]},
        sim_fn=lambda v1, v2: 0.95,
    )
    aligned, score = dc.is_domain_aligned(prompt)
    assert aligned is True
    assert score == pytest.approx(0.95)


def test_is_domain_aligned_out_of_domain_prompt_scores_below_threshold(monkeypatch, tmp_path):
    corpus_docs = ["How does a neural network backpropagate gradients?"]
    prompt = "What's a good recipe for baking sourdough bread this weekend?"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)
    _patch_embeddings(
        monkeypatch,
        corpus_vec_map={corpus_docs[0]: [1.0, 0.0]},
        prompt_vec_map={prompt: [0.0, 1.0]},
        sim_fn=lambda v1, v2: 0.03,
    )
    aligned, score = dc.is_domain_aligned(prompt)
    assert aligned is False
    assert score == pytest.approx(0.03)


def test_is_domain_aligned_returns_tuple_of_bool_and_float(monkeypatch, tmp_path):
    corpus_docs = ["Explain SQL injection defenses."]
    prompt = "How do prepared statements prevent SQL injection?"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)
    _patch_embeddings(
        monkeypatch,
        corpus_vec_map={corpus_docs[0]: [1.0, 0.0]},
        prompt_vec_map={prompt: [1.0, 0.0]},
        sim_fn=lambda v1, v2: 0.5,
    )
    result = dc.is_domain_aligned(prompt)
    assert isinstance(result, tuple)
    assert len(result) == 2
    is_aligned, score = result
    assert isinstance(is_aligned, bool)
    assert isinstance(score, float)


@pytest.mark.parametrize("threshold_score", [0.22, 0.5, 1.0])
def test_is_domain_aligned_boundary_at_or_above_threshold_is_aligned(monkeypatch, tmp_path, threshold_score):
    corpus_docs = ["technical doc"]
    prompt = "some prompt"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)
    _patch_embeddings(
        monkeypatch,
        corpus_vec_map={corpus_docs[0]: [1.0]},
        prompt_vec_map={prompt: [1.0]},
        sim_fn=lambda v1, v2: threshold_score,
    )
    aligned, score = dc.is_domain_aligned(prompt)
    assert aligned is True
    assert score == pytest.approx(threshold_score)


def test_is_domain_aligned_just_below_threshold_is_not_aligned(monkeypatch, tmp_path):
    corpus_docs = ["technical doc"]
    prompt = "some prompt"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)
    _patch_embeddings(
        monkeypatch,
        corpus_vec_map={corpus_docs[0]: [1.0]},
        prompt_vec_map={prompt: [1.0]},
        sim_fn=lambda v1, v2: 0.2199,
    )
    aligned, score = dc.is_domain_aligned(prompt)
    assert aligned is False


def test_is_domain_aligned_defaults_to_allow_when_corpus_missing(monkeypatch, tmp_path, caplog):
    """If the corpus file / centroid is unavailable, the classifier must
    default to allow (True, 1.0) rather than block or crash — this is a
    documented fail-open policy in _get_domain_centroid."""
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(dc, "DOMAIN_CORPUS_FILE", str(missing))
    with caplog.at_level(logging.WARNING):
        aligned, score = dc.is_domain_aligned("Anything at all, even off-topic text.")
    assert aligned is True
    assert score == 1.0
    assert any("Domain centroid not computed" in r.message for r in caplog.records)


def test_is_domain_aligned_defaults_to_allow_when_prompt_embedding_fails(monkeypatch, tmp_path):
    corpus_docs = ["technical doc"]
    prompt = "some prompt"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)

    def fake_get_embedding(text):
        if text == corpus_docs[0]:
            return [1.0, 0.0]
        return None  # prompt embedding fails

    monkeypatch.setattr("core.embeddings.get_embedding", fake_get_embedding)
    monkeypatch.setattr("core.embeddings.cosine_similarity", lambda v1, v2: 0.0)

    aligned, score = dc.is_domain_aligned(prompt)
    assert aligned is True
    assert score == 1.0


def test_is_domain_aligned_empty_string_prompt_does_not_crash(monkeypatch, tmp_path):
    corpus_docs = ["technical doc"]
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)
    _patch_embeddings(
        monkeypatch,
        corpus_vec_map={corpus_docs[0]: [1.0]},
        prompt_vec_map={"": [0.0]},
        sim_fn=lambda v1, v2: 0.01,
    )
    aligned, score = dc.is_domain_aligned("")
    assert aligned is False
    assert isinstance(score, float)


def test_is_domain_aligned_converts_tensor_like_prompt_vector(monkeypatch, tmp_path):
    """Prompt embeddings may also come back as tensor-like objects with
    .tolist() — same conversion path as corpus documents."""

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    corpus_docs = ["technical doc"]
    prompt = "some prompt"
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)

    seen = {}

    def fake_get_embedding(text):
        if text == corpus_docs[0]:
            return [1.0, 0.0]
        return FakeTensor([1.0, 0.0])

    def fake_cosine(v1, v2):
        seen["v1"] = v1
        return 0.9

    monkeypatch.setattr("core.embeddings.get_embedding", fake_get_embedding)
    monkeypatch.setattr("core.embeddings.cosine_similarity", fake_cosine)

    aligned, score = dc.is_domain_aligned(prompt)
    assert aligned is True
    assert score == 0.9
    # cosine_similarity must have received a plain list, not the tensor-like object.
    assert seen["v1"] == [1.0, 0.0]


def test_is_domain_aligned_uses_cached_centroid_across_calls(monkeypatch, tmp_path):
    """The centroid should only be (re)computed once even across multiple
    is_domain_aligned calls."""
    corpus_docs = ["technical doc"]
    _setup_corpus(monkeypatch, tmp_path, corpus_docs)

    embed_calls = []

    def fake_get_embedding(text):
        embed_calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr("core.embeddings.get_embedding", fake_get_embedding)
    monkeypatch.setattr("core.embeddings.cosine_similarity", lambda v1, v2: 0.5)

    dc.is_domain_aligned("first prompt")
    dc.is_domain_aligned("second prompt")

    # Corpus doc embedded exactly once; each prompt embedded once each call.
    assert embed_calls.count(corpus_docs[0]) == 1
    assert embed_calls.count("first prompt") == 1
    assert embed_calls.count("second prompt") == 1
