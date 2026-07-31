"""
Tests for the pluggable detector layer.

Models are mocked throughout: these tests verify the wiring, not the classifiers.
The wiring is where the dangerous bugs live — a detector with inverted labels
still returns well-formed probabilities and still produces a plausible results
table, it is just backwards. That failure is silent, so it gets tests.
"""
import types

import pytest

from core.detectors import (
    CLASS_HARMFUL,
    CLASS_INJECTION,
    CLASS_JAILBREAK,
    AnchorDetector,
    Detector,
    TransformerDetector,
    available_detectors,
    get_detector,
    get_registry,
)


# --- registry ---------------------------------------------------------------

def test_registry_contains_baseline_and_public_detectors():
    reg = get_registry()
    assert "anchors" in reg
    assert isinstance(reg["anchors"], AnchorDetector)
    assert "protectai_injection" in reg
    assert "toxic_bert" in reg


def test_every_detector_declares_targets_and_description():
    for name, detector in get_registry().items():
        assert detector.targets, f"{name} declares no target classes"
        assert detector.description, f"{name} has no description"
        assert detector.name == name, f"{name} has mismatched .name"


def test_all_three_attack_classes_are_covered_by_some_detector():
    """
    Measurement showed the anchor detector catches only 24.4% of harmful
    content. If no detector targets a class, that class is unmonitored.
    """
    covered = set()
    for detector in get_registry().values():
        covered.update(detector.targets)
    assert {CLASS_INJECTION, CLASS_JAILBREAK, CLASS_HARMFUL} <= covered


def test_contaminated_detectors_declare_their_training_sources():
    """
    Detectors trained on suite sources MUST declare them or the comparison
    silently rewards memorisation.
    """
    reg = get_registry()
    assert "jackhhao/jailbreak-classification" in reg["jailbreak_classifier"].trained_on
    assert "deepset/prompt-injections" in reg["deepset_injection"].trained_on
    # The baseline is not fitted on anything.
    assert reg["anchors"].trained_on == ()


def test_get_detector_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown detector"):
        get_detector("does_not_exist")


# --- TransformerDetector label resolution -----------------------------------

def _fake_transformer(detector, id2label, logits_by_text, multi_label=False):
    """Installs a stub model/tokenizer so no download or inference occurs."""
    import torch

    detector._tokenizer = lambda texts, **kw: {"input_ids": texts}
    detector._model = types.SimpleNamespace(
        config=types.SimpleNamespace(id2label=id2label)
    )
    detector._positive_ids = [
        i for i, label in id2label.items()
        if str(label).lower() in detector.positive_labels
    ]

    def forward(**kwargs):
        rows = [logits_by_text[t] for t in kwargs["input_ids"]]
        return types.SimpleNamespace(logits=torch.tensor(rows))

    detector._model.__call__ = forward
    # torch.no_grad + model(**enc) path needs the object itself to be callable.
    detector._model = types.SimpleNamespace(
        config=detector._model.config, __call__=forward
    )

    class Callable:
        def __init__(self, cfg, fn):
            self.config = cfg
            self._fn = fn

        def __call__(self, **kw):
            return self._fn(**kw)

    detector._model = Callable(types.SimpleNamespace(id2label=id2label), forward)
    detector._load = lambda: None
    return detector


def test_positive_label_resolved_by_name_not_index():
    """
    Two models with the SAME labels in OPPOSITE order must produce the same
    score. Hard-coding index 1 as "attack" would invert one of them.
    """
    import torch  # noqa: F401

    logits = {"attack text": [0.0, 5.0], "benign text": [5.0, 0.0]}
    d1 = TransformerDetector("d1", "fake/m", ["injection"], (CLASS_INJECTION,))
    _fake_transformer(d1, {0: "SAFE", 1: "INJECTION"}, logits)

    flipped = {"attack text": [5.0, 0.0], "benign text": [0.0, 5.0]}
    d2 = TransformerDetector("d2", "fake/m", ["injection"], (CLASS_INJECTION,))
    _fake_transformer(d2, {0: "INJECTION", 1: "SAFE"}, flipped)

    a1, b1 = d1.score_batch(["attack text", "benign text"])
    a2, b2 = d2.score_batch(["attack text", "benign text"])

    assert a1 > b1 and a2 > b2
    assert a1 == pytest.approx(a2, abs=1e-6)


def test_label_matching_is_case_insensitive():
    d = TransformerDetector("d", "fake/m", ["INJECTION"], (CLASS_INJECTION,))
    _fake_transformer(d, {0: "safe", 1: "injection"},
                      {"x": [0.0, 3.0]})
    assert d.score_batch(["x"])[0] > 0.9


def test_multi_label_uses_sigmoid_not_softmax():
    """
    Toxicity models emit independent per-label probabilities. Softmax would
    force them to sum to 1 and distort every score.
    """
    logits = {"x": [3.0, 3.0, -5.0, -5.0]}
    id2label = {0: "toxic", 1: "insult", 2: "threat", 3: "obscene"}

    sig = TransformerDetector("sig", "fake/m", ["toxic"], (CLASS_HARMFUL,), multi_label=True)
    _fake_transformer(sig, id2label, logits)
    soft = TransformerDetector("soft", "fake/m", ["toxic"], (CLASS_HARMFUL,), multi_label=False)
    _fake_transformer(soft, id2label, logits)

    # sigmoid(3.0) ~= 0.953; softmax over two tied highs ~= 0.5
    assert sig.score_batch(["x"])[0] == pytest.approx(0.9526, abs=1e-3)
    assert soft.score_batch(["x"])[0] == pytest.approx(0.5, abs=1e-2)


def test_multi_label_takes_max_across_positive_labels():
    logits = {"x": [-5.0, 4.0, -5.0]}
    d = TransformerDetector("d", "fake/m", ["toxic", "insult"], (CLASS_HARMFUL,),
                            multi_label=True)
    _fake_transformer(d, {0: "toxic", 1: "insult", 2: "threat"}, logits)
    # 'insult' is the strong one and must win.
    assert d.score_batch(["x"])[0] == pytest.approx(0.982, abs=1e-2)


def test_scores_are_bounded_probabilities():
    d = TransformerDetector("d", "fake/m", ["injection"], (CLASS_INJECTION,))
    _fake_transformer(d, {0: "safe", 1: "injection"},
                      {"a": [-20.0, 20.0], "b": [20.0, -20.0]})
    for s in d.score_batch(["a", "b"]):
        assert 0.0 <= s <= 1.0


# --- failure handling -------------------------------------------------------

def test_unresolvable_labels_make_detector_unavailable_not_wrong():
    """
    A label mismatch must disable the detector, never fall back to a guess.
    Guessing produces a detector that is confidently backwards.
    """
    d = TransformerDetector("d", "fake/m", ["nonexistent_label"], (CLASS_INJECTION,))
    d._tokenizer = object()
    d._positive_ids = []
    d._load_error = "none of ['nonexistent_label'] found in model labels ['safe', 'injection']"

    ok, detail = d.available()
    assert ok is False
    assert "nonexistent_label" in detail

    with pytest.raises(RuntimeError, match="unavailable"):
        d.score_batch(["anything"])


def test_available_detectors_partitions_into_ok_and_bad(monkeypatch):
    good = AnchorDetector()
    monkeypatch.setattr(good, "available", lambda: (True, "fine"))
    bad = AnchorDetector()
    monkeypatch.setattr(bad, "available", lambda: (False, "broken"))
    monkeypatch.setattr("core.detectors._REGISTRY", {"good": good, "bad": bad})

    ok, unavailable = available_detectors()
    assert ("good", "fine") in ok
    assert ("bad", "broken") in unavailable


def test_base_detector_score_delegates_to_batch():
    class Stub(Detector):
        name = "stub"
        targets = (CLASS_INJECTION,)

        def score_batch(self, texts):
            return [0.42] * len(texts)

    assert Stub().score("anything") == 0.42
