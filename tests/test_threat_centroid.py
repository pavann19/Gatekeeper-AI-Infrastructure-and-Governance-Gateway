"""
Tests for core/threat_centroid.py — centroid-based malicious intent
detection used by core.risk.compute_centroid_similarity (via
`signals["centroid_score"]`).

The module lazily loads threat anchors from policies.json (cwd-relative),
builds a centroid by averaging their embeddings (core.embeddings.
get_embedding), then scores new prompts by cosine similarity to that
centroid. Embeddings and cosine_similarity are monkeypatched with tiny
deterministic fakes so tests are fast and never touch sentence-transformers,
matching the discipline in tests/test_embedding_head_detector.py.

Because the module keeps process-global lazy-init state
(_threat_centroid_cache / _threat_centroid_initialized), every test resets
that state before running so tests don't leak into each other.
"""
import json
import logging

import pytest

import core.threat_centroid as tc


@pytest.fixture(autouse=True)
def reset_centroid_cache(monkeypatch, tmp_path):
    """Reset lazy-init globals and run each test in an isolated cwd so
    POLICY_FILE ("policies.json", read relative to cwd) doesn't touch the
    real project policies.json."""
    tc._threat_centroid_cache = None
    tc._threat_centroid_initialized = False
    monkeypatch.chdir(tmp_path)
    yield
    tc._threat_centroid_cache = None
    tc._threat_centroid_initialized = False


def _write_policies(classes=None, flat_anchors=None):
    data = {}
    if classes is not None:
        data["threat_anchor_classes"] = classes
    if flat_anchors is not None:
        data["threat_anchors"] = flat_anchors
    with open("policies.json", "w") as f:
        json.dump(data, f)


def _fake_get_embedding(vectors_by_text):
    def fake(text):
        return list(vectors_by_text[text])
    return fake


def _install_fake_embeddings(monkeypatch, vectors_by_text, cosine_fn=None):
    monkeypatch.setattr(
        "core.embeddings.get_embedding", _fake_get_embedding(vectors_by_text)
    )
    if cosine_fn is not None:
        monkeypatch.setattr("core.embeddings.cosine_similarity", cosine_fn)


# --- load_threat_anchors -----------------------------------------------------

def test_load_anchors_missing_file_returns_empty_and_warns(caplog):
    caplog.set_level(logging.WARNING)
    anchors = tc.load_threat_anchors()
    assert anchors == []
    assert "not found" in caplog.text


def test_load_anchors_prefers_grouped_classes_over_flat():
    _write_policies(
        classes={"injection": ["ignore instructions"], "exfil": ["dump the database"]},
        flat_anchors=["this flat list must be ignored"],
    )
    anchors = tc.load_threat_anchors()
    assert set(anchors) == {"ignore instructions", "dump the database"}
    assert "this flat list must be ignored" not in anchors


def test_load_anchors_falls_back_to_flat_when_no_classes():
    _write_policies(flat_anchors=["do something bad"])
    anchors = tc.load_threat_anchors()
    assert anchors == ["do something bad"]


def test_load_anchors_empty_classes_dict_falls_back_to_flat():
    # {} is falsy, so `if classes:` should fall through to the flat key.
    _write_policies(classes={}, flat_anchors=["fallback anchor"])
    anchors = tc.load_threat_anchors()
    assert anchors == ["fallback anchor"]


def test_load_anchors_no_keys_present_returns_empty():
    _write_policies()
    anchors = tc.load_threat_anchors()
    assert anchors == []


def test_load_anchors_logs_count_on_success(caplog):
    caplog.set_level(logging.INFO)
    _write_policies(classes={"a": ["one", "two"]})
    tc.load_threat_anchors()
    assert "Loaded 2 threat anchors" in caplog.text


def test_load_anchors_corrupt_json_returns_empty_and_warns(caplog):
    caplog.set_level(logging.WARNING)
    with open("policies.json", "w") as f:
        f.write("{ not valid json")
    anchors = tc.load_threat_anchors()
    assert anchors == []
    assert "Failed to load threat anchors" in caplog.text


# --- build_malicious_centroid ------------------------------------------------

def test_build_centroid_no_anchors_returns_none():
    assert tc.build_malicious_centroid([]) is None


def test_build_centroid_is_elementwise_average(monkeypatch):
    _install_fake_embeddings(monkeypatch, {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
    })
    centroid = tc.build_malicious_centroid(["a", "b"])
    assert centroid == pytest.approx([0.5, 0.5, 0.0])


def test_build_centroid_skips_none_embeddings(monkeypatch):
    def fake(text):
        return None if text == "bad" else [2.0, 2.0]
    monkeypatch.setattr("core.embeddings.get_embedding", fake)
    centroid = tc.build_malicious_centroid(["bad", "good"])
    assert centroid == pytest.approx([2.0, 2.0])


def test_build_centroid_all_none_returns_none(monkeypatch):
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: None)
    assert tc.build_malicious_centroid(["a", "b"]) is None


def test_build_centroid_handles_tensor_like_objects_via_tolist(monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    monkeypatch.setattr(
        "core.embeddings.get_embedding",
        lambda text: FakeTensor([1.0, 3.0]),
    )
    centroid = tc.build_malicious_centroid(["x"])
    assert centroid == pytest.approx([1.0, 3.0])


def test_build_centroid_logs_dimension_and_count(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _install_fake_embeddings(monkeypatch, {"a": [1.0, 2.0, 3.0]})
    tc.build_malicious_centroid(["a"])
    assert "1 vectors" in caplog.text
    assert "dim=3" in caplog.text


# --- lazy singleton behavior (_get_malicious_centroid) -----------------------

def test_centroid_is_computed_once_and_cached(monkeypatch):
    _write_policies(classes={"c": ["anchor one"]})
    calls = []

    def fake(text):
        calls.append(text)
        return [1.0, 1.0]

    monkeypatch.setattr("core.embeddings.get_embedding", fake)

    first = tc._get_malicious_centroid()
    second = tc._get_malicious_centroid()

    assert first == pytest.approx([1.0, 1.0])
    assert first is second or first == second
    # embeddings computed only on the first call, not the second
    assert calls == ["anchor one"]


def test_centroid_cache_persists_even_if_policy_file_changes_after_init(monkeypatch):
    _write_policies(classes={"c": ["anchor one"]})
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: [1.0, 0.0])

    first = tc._get_malicious_centroid()
    assert first == pytest.approx([1.0, 0.0])

    # Rewrite policies.json — the cached centroid must not change because
    # initialization already happened (lazy-once semantics).
    _write_policies(classes={"c": ["a totally different anchor"]})
    second = tc._get_malicious_centroid()
    assert second == first


def test_missing_policy_file_yields_none_centroid_without_raising(caplog):
    caplog.set_level(logging.WARNING)
    centroid = tc._get_malicious_centroid()
    assert centroid is None
    assert "disabled" in caplog.text.lower()


# --- compute_centroid_similarity --------------------------------------------

def test_similarity_returns_zero_when_centroid_unavailable():
    # No policies.json in this cwd -> anchors empty -> centroid None.
    assert tc.compute_centroid_similarity([1.0, 2.0, 3.0]) == 0.0


def test_similarity_returns_zero_when_prompt_vec_is_none(monkeypatch):
    _write_policies(classes={"c": ["anchor"]})
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: [1.0, 0.0])
    assert tc.compute_centroid_similarity(None) == 0.0


def test_similarity_of_identical_vector_to_centroid_is_one(monkeypatch):
    _write_policies(classes={"c": ["anchor"]})
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: [1.0, 0.0])

    score = tc.compute_centroid_similarity([1.0, 0.0])
    assert score == pytest.approx(1.0)


def test_similarity_of_orthogonal_vector_to_centroid_is_zero(monkeypatch):
    _write_policies(classes={"c": ["anchor"]})
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: [1.0, 0.0])

    score = tc.compute_centroid_similarity([0.0, 1.0])
    assert score == pytest.approx(0.0, abs=1e-9)


def test_similarity_of_opposite_vector_to_centroid_is_negative_one(monkeypatch):
    _write_policies(classes={"c": ["anchor"]})
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: [1.0, 0.0])

    score = tc.compute_centroid_similarity([-1.0, 0.0])
    assert score == pytest.approx(-1.0)


def test_similarity_handles_tensor_like_prompt_vec_via_tolist(monkeypatch):
    _write_policies(classes={"c": ["anchor"]})
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: [1.0, 0.0])

    class FakeTensor:
        def tolist(self):
            return [1.0, 0.0]

    score = tc.compute_centroid_similarity(FakeTensor())
    assert score == pytest.approx(1.0)


def test_similarity_averages_multiple_anchors_correctly(monkeypatch):
    # Two anchors at [1,0] and [0,1] -> centroid [0.5,0.5] (not normalized).
    # Cosine similarity of [1,0] against [0.5,0.5] is 1/sqrt(2).
    _write_policies(classes={"c": ["anchor_a", "anchor_b"]})
    vectors = {"anchor_a": [1.0, 0.0], "anchor_b": [0.0, 1.0]}
    monkeypatch.setattr("core.embeddings.get_embedding", lambda text: vectors[text])

    score = tc.compute_centroid_similarity([1.0, 0.0])
    assert score == pytest.approx(0.7071067811865476, abs=1e-6)
