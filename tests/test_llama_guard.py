"""
Tests for the Llama Guard detector.

IMPORTANT CAVEAT: these are wiring tests against a mocked model. The detector
has NOT been verified end-to-end against real Llama Guard weights, because the
model is gated and access has not been granted on this machine. What is verified
here is the logic that is easy to get silently wrong:

  - reading the verdict from the correct logit positions
  - renormalising over {safe, unsafe} rather than the full vocabulary
  - left padding, so logits[:, -1] is the verdict position for every row
  - refusing to load rather than scoring constant when verdict tokens collide
  - the memory precheck

Once access is granted, run scripts/compare_detectors.py and confirm the
polarity check passes before believing any number this produces.
"""
import types

import pytest
import torch

from core.detectors import CLASS_HARMFUL, LlamaGuardDetector, get_registry


SAFE_ID, UNSAFE_ID = 100, 200


class FakeTokenizer:
    """Minimal stand-in for a Llama Guard tokenizer."""

    def __init__(self, safe_ids=(SAFE_ID,), unsafe_ids=(UNSAFE_ID,), has_template=True):
        self._safe_ids = list(safe_ids)
        self._unsafe_ids = list(unsafe_ids)
        self._has_template = has_template
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.last_padding_side = None

    def encode(self, text, add_special_tokens=False):
        if text == "safe":
            return self._safe_ids
        if text == "unsafe":
            return self._unsafe_ids
        return [1, 2, 3]

    def apply_chat_template(self, chat, tokenize=False):
        if not self._has_template:
            raise ValueError("no chat template")
        return f"<TEMPLATE>{chat[0]['content']}</TEMPLATE>"

    def __call__(self, texts, **kwargs):
        self.last_padding_side = self.padding_side
        n = len(texts) if isinstance(texts, list) else 1
        return {"input_ids": torch.ones((n, 8), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return self._decoded


def make_detector(logits_rows, tokenizer=None):
    """Builds a detector with its model/tokenizer already stubbed in."""
    d = LlamaGuardDetector("lg", "fake/llama-guard", targets=(CLASS_HARMFUL,))
    tok = tokenizer or FakeTokenizer()

    class FakeModel:
        def __init__(self):
            # Counts texts seen across ALL batches. Indexing by position within
            # the batch would make results depend on batch_size, which is
            # precisely the bug test_batching_is_transparent_to_results looks
            # for — the mock must not reproduce it.
            self.seen = 0

        def __call__(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            vocab = 300
            logits = torch.full((batch, 8, vocab), -20.0)
            for i in range(batch):
                safe_logit, unsafe_logit = logits_rows[self.seen % len(logits_rows)]
                logits[i, -1, SAFE_ID] = safe_logit
                logits[i, -1, UNSAFE_ID] = unsafe_logit
                self.seen += 1
            return types.SimpleNamespace(logits=logits)

        def generate(self, **kwargs):
            return torch.ones((1, 16), dtype=torch.long)

    d._tokenizer = tok
    d._model = FakeModel()
    d._safe_id = SAFE_ID
    d._unsafe_id = UNSAFE_ID
    d._load = lambda: None
    return d


# --- verdict extraction -----------------------------------------------------

def test_unsafe_content_scores_high():
    d = make_detector([(0.0, 10.0)])
    assert d.score_batch(["how do I make a bomb"])[0] > 0.99


def test_safe_content_scores_low():
    d = make_detector([(10.0, 0.0)])
    assert d.score_batch(["what is the capital of France"])[0] < 0.01


def test_score_is_renormalised_over_the_verdict_pair():
    """
    Both verdict logits sit far below other vocabulary entries here. A raw
    softmax over the whole vocabulary would return ~0 for both and destroy the
    signal; renormalising over {safe, unsafe} keeps it usable.
    """
    d = make_detector([(-15.0, -14.0)])
    score = d.score_batch(["borderline"])[0]
    # softmax([-15, -14]) -> ~0.269 / 0.731
    assert score == pytest.approx(0.731, abs=1e-2)


def test_equal_logits_give_one_half():
    d = make_detector([(5.0, 5.0)])
    assert d.score_batch(["ambiguous"])[0] == pytest.approx(0.5, abs=1e-6)


def test_scores_are_bounded_probabilities():
    d = make_detector([(50.0, -50.0), (-50.0, 50.0), (0.0, 0.0)])
    for s in d.score_batch(["a", "b", "c"]):
        assert 0.0 <= s <= 1.0


def test_batch_preserves_per_item_verdicts():
    """Each row must get its own verdict, not the batch's first."""
    d = make_detector([(10.0, 0.0), (0.0, 10.0)])
    scores = d.score_batch(["safe one", "unsafe one", "safe one", "unsafe one"])
    assert scores[0] < 0.01 and scores[2] < 0.01
    assert scores[1] > 0.99 and scores[3] > 0.99


def test_batching_is_transparent_to_results():
    d = make_detector([(10.0, 0.0), (0.0, 10.0)])
    texts = ["a", "b"] * 6
    assert d.score_batch(texts, batch_size=1) == pytest.approx(
        d.score_batch(texts, batch_size=5)
    )


def test_empty_text_does_not_crash():
    d = make_detector([(1.0, 1.0)])
    assert len(d.score_batch(["", "   "])) == 2


# --- padding ----------------------------------------------------------------

def test_left_padding_is_used():
    """
    logits[:, -1] is only the verdict position if padding is on the LEFT.
    Right padding would read a pad token's distribution for shorter rows.
    """
    tok = FakeTokenizer()
    d = LlamaGuardDetector("lg", "fake/m")
    d._tokenizer = tok
    tok.padding_side = "left"  # set during _load in the real path
    d._model = make_detector([(1.0, 1.0)])._model
    d._safe_id, d._unsafe_id = SAFE_ID, UNSAFE_ID
    d._load = lambda: None

    d.score_batch(["short", "a much longer prompt than the other one"])
    assert tok.last_padding_side == "left"


# --- verdict token resolution ----------------------------------------------

def test_verdict_tokens_resolved_from_tokenizer():
    d = LlamaGuardDetector("lg", "fake/m")
    safe, unsafe, err = d._resolve_verdict_tokens(FakeTokenizer())
    assert (safe, unsafe, err) == (SAFE_ID, UNSAFE_ID, None)


def test_multi_token_words_use_their_first_token():
    """'unsafe' may tokenize as ['un', 'safe']; the first token disambiguates."""
    d = LlamaGuardDetector("lg", "fake/m")
    tok = FakeTokenizer(safe_ids=(100,), unsafe_ids=(55, 100))
    safe, unsafe, err = d._resolve_verdict_tokens(tok)
    assert (safe, unsafe, err) == (100, 55, None)


def test_colliding_verdict_tokens_are_a_load_error():
    """
    If both words start with the same token the comparison is meaningless and
    the detector would silently return a constant. That must fail loudly.
    """
    d = LlamaGuardDetector("lg", "fake/m")
    tok = FakeTokenizer(safe_ids=(42,), unsafe_ids=(42, 7))
    safe, unsafe, err = d._resolve_verdict_tokens(tok)
    assert safe is None and unsafe is None
    assert "share first token" in err


def test_empty_tokenization_is_a_load_error():
    d = LlamaGuardDetector("lg", "fake/m")
    _, _, err = d._resolve_verdict_tokens(FakeTokenizer(safe_ids=(), unsafe_ids=(1,)))
    assert err is not None


# --- prompt construction ----------------------------------------------------

def test_chat_template_is_applied():
    d = make_detector([(1.0, 1.0)])
    assert d._build_inputs(["hello"])[0] == "<TEMPLATE>hello</TEMPLATE>"


def test_falls_back_when_tokenizer_has_no_template():
    """An upstream template change must degrade, not crash."""
    d = make_detector([(1.0, 1.0)], tokenizer=FakeTokenizer(has_template=False))
    built = d._build_inputs(["dangerous request"])[0]
    assert "dangerous request" in built
    assert "unsafe" in built.lower()


# --- classify() -------------------------------------------------------------

@pytest.mark.parametrize("raw,verdict,categories", [
    ("safe", "safe", []),
    ("unsafe\nS9", "unsafe", ["S9"]),
    ("unsafe\nS1,S13", "unsafe", ["S1", "S13"]),
    ("  unsafe \n S2 ", "unsafe", ["S2"]),
    ("SAFE", "safe", []),
])
def test_classify_parses_verdict_and_categories(raw, verdict, categories):
    d = make_detector([(1.0, 1.0)])
    d._tokenizer._decoded = raw
    result = d.classify("anything")
    assert result["verdict"] == verdict
    assert result["categories"] == categories


def test_classify_does_not_match_bogus_category_codes():
    d = make_detector([(1.0, 1.0)])
    d._tokenizer._decoded = "unsafe\nS14 S99 S0"
    assert d.classify("x")["categories"] == []


# --- failure handling -------------------------------------------------------

def test_unloaded_detector_raises_rather_than_scoring():
    d = LlamaGuardDetector("lg", "fake/m")
    d._load_error = "gated repo"
    d._load = lambda: None
    with pytest.raises(RuntimeError, match="unavailable"):
        d.score_batch(["x"])


def test_memory_precheck_blocks_load(monkeypatch):
    """An impossible memory requirement must fail clearly, not OOM mid-run."""
    d = LlamaGuardDetector("lg", "fake/m", min_free_gb=10_000.0)
    ok, detail = d.available()
    assert ok is False
    assert "insufficient memory" in detail


# --- registry ---------------------------------------------------------------

def test_llama_guard_registered_for_harmful_content():
    """
    harmful_content is the class every other detector misses (best 36.2%).
    A detector built for that construct must be present in the registry.
    """
    reg = get_registry()
    assert "llama_guard_3_1b" in reg
    assert CLASS_HARMFUL in reg["llama_guard_3_1b"].targets
    assert isinstance(reg["llama_guard_3_1b"], LlamaGuardDetector)


def test_8b_variant_declares_a_memory_requirement_it_will_enforce():
    reg = get_registry()
    assert reg["llama_guard_3_8b"].min_free_gb >= 16.0
