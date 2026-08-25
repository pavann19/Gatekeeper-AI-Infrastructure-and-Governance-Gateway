"""
New coverage for core/output_guardrails.py not already exercised by
tests/test_output_guardrails.py: check_semantic_grounding's own boundary
behaviour, check_system_prompt_leakage's exact min_run boundary, the
hallucination BLOCK path end-to-end, and ordering/priority combinations
between findings that the existing file doesn't combine.
"""
from unittest.mock import patch

import pytest

from core.output_guardrails import (
    assess_output,
    check_semantic_grounding,
    check_system_prompt_leakage,
)


# ---------------------------------------------------------------------------
# check_semantic_grounding -- centroid-availability and threshold boundary
# ---------------------------------------------------------------------------

def test_grounding_returns_true_when_centroid_unavailable():
    """Not enough educational-domain data yet -- fail open, not closed."""
    with patch("core.output_guardrails.educational_store") as store:
        store.get_centroid.return_value = None
        assert check_semantic_grounding("anything at all") is True


def test_grounding_false_when_score_below_threshold():
    with patch("core.output_guardrails.educational_store") as store, \
         patch("core.output_guardrails.get_embedding", return_value=[1.0, 0.0]), \
         patch("core.output_guardrails.cosine_similarity", return_value=0.10):
        store.get_centroid.return_value = [0.0, 1.0]
        assert check_semantic_grounding("orthogonal text") is False


def test_grounding_true_when_score_above_threshold():
    with patch("core.output_guardrails.educational_store") as store, \
         patch("core.output_guardrails.get_embedding", return_value=[1.0, 0.0]), \
         patch("core.output_guardrails.cosine_similarity", return_value=0.20):
        store.get_centroid.return_value = [1.0, 0.0]
        assert check_semantic_grounding("aligned text") is True


def test_grounding_exact_threshold_boundary_is_inclusive_of_pass():
    """Score == 0.15 must NOT be flagged (the check is `< 0.15`, strictly less)."""
    with patch("core.output_guardrails.educational_store") as store, \
         patch("core.output_guardrails.get_embedding", return_value=[1.0]), \
         patch("core.output_guardrails.cosine_similarity", return_value=0.15):
        store.get_centroid.return_value = [1.0]
        assert check_semantic_grounding("boundary text") is True


def test_grounding_just_under_threshold_is_flagged():
    with patch("core.output_guardrails.educational_store") as store, \
         patch("core.output_guardrails.get_embedding", return_value=[1.0]), \
         patch("core.output_guardrails.cosine_similarity", return_value=0.1499):
        store.get_centroid.return_value = [1.0]
        assert check_semantic_grounding("just under") is False


# ---------------------------------------------------------------------------
# check_system_prompt_leakage -- exact min_run boundary
# ---------------------------------------------------------------------------

def test_leakage_run_exactly_at_min_run_is_detected():
    run = "x" * 40
    system_prompt = f"prefix {run} suffix"
    response = f"here is {run} embedded"
    assert check_system_prompt_leakage(response, system_prompt, min_run=40) is True


def test_leakage_run_one_under_min_run_is_not_detected():
    """39 identical characters bracketed by DIFFERENT text on each side, so
    no 40-char sliding window from system_prompt (including ones that eat
    into the bracketing text) can align with a 40-char window in response."""
    run = "x" * 39
    system_prompt = f"AAAAA{run}BBBBB"
    response = f"CCCCC{run}DDDDD"
    assert check_system_prompt_leakage(response, system_prompt, min_run=40) is False


def test_leakage_system_prompt_shorter_than_min_run_never_matches():
    short_prompt = "y" * 39
    assert check_system_prompt_leakage(short_prompt, short_prompt, min_run=40) is False


def test_leakage_custom_min_run_is_respected():
    system_prompt = "abcdefghij"
    response = "prefix abcdefghij suffix"
    assert check_system_prompt_leakage(response, system_prompt, min_run=10) is True
    assert check_system_prompt_leakage(response, system_prompt, min_run=11) is False


def test_leakage_case_sensitive_verbatim_match():
    system_prompt = "X" * 40
    response = ("x" * 40)  # different case, same length
    assert check_system_prompt_leakage(response, system_prompt, min_run=40) is False


# ---------------------------------------------------------------------------
# assess_output -- hallucination path end-to-end
# ---------------------------------------------------------------------------

def test_assess_output_blocks_on_hallucination_when_everything_else_passes():
    """HALLUCINATION_CHECK_ENABLED defaults False (see core/config.py's
    docstring -- a live pentest found this check compares output against
    the wrong reference corpus). This test explicitly enables it to prove
    the gated code path itself still works correctly when turned on."""
    with patch("core.output_guardrails.settings.HALLUCINATION_CHECK_ENABLED", True), \
         patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=False):
        decision, details = assess_output("a response about an unrelated domain")
        assert decision == "BLOCK"
        assert details["hallucination_detected"] is True
        assert details["source"] == "semantic_grounding_check"
        assert details["clean_response"] is None


def test_assess_output_hallucination_check_skipped_by_default():
    """The regression this whole gate exists for: with the flag at its
    real default (False), a response that WOULD have failed grounding must
    still ALLOW, because the check never runs at all."""
    with patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=False) as mock_grounding:
        decision, details = assess_output("a response about an unrelated domain")
        assert decision == "ALLOW"
        assert details["hallucination_detected"] is False
        mock_grounding.assert_not_called()


def test_assess_output_grounding_check_runs_on_redacted_text():
    """Same discipline as the toxicity judge -- grounding must never see
    raw PII the redaction step already removed."""
    seen = {}

    def _fake_grounding(text):
        seen["text"] = text
        return True

    with patch("core.output_guardrails.settings.HALLUCINATION_CHECK_ENABLED", True), \
         patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", side_effect=_fake_grounding):
        assess_output("Contact john.doe@example.com for the report.")
        assert "john.doe@example.com" not in seen["text"]
        assert "[REDACTED:EMAIL]" in seen["text"]


def test_assess_output_grounding_not_evaluated_if_toxicity_already_blocked():
    with patch("core.output_guardrails.output_judge", return_value="DANGEROUS"), \
         patch("core.output_guardrails.check_semantic_grounding") as mock_grounding:
        decision, details = assess_output("harmful content")
        assert decision == "BLOCK"
        assert details["toxicity_detected"] is True
        mock_grounding.assert_not_called()


# ---------------------------------------------------------------------------
# Ordering / priority combinations not covered by the existing file
# ---------------------------------------------------------------------------

def test_secret_takes_priority_over_system_prompt_leakage():
    system_prompt = "z" * 50
    response_text = f"key: AKIAABCDEFGHIJKLMNOP and also {system_prompt} leaked"
    decision, details = assess_output(response_text, system_prompt=system_prompt)
    assert decision == "BLOCK"
    assert details["secrets_detected"] is True
    assert details["system_prompt_leak_detected"] is False  # never reached


def test_system_prompt_leakage_takes_priority_over_pii_and_toxicity():
    system_prompt = "w" * 50
    response_text = f"{system_prompt} contact john.doe@example.com"
    with patch("core.output_guardrails.output_judge") as mock_judge:
        decision, details = assess_output(response_text, system_prompt=system_prompt)
        assert decision == "BLOCK"
        assert details["system_prompt_leak_detected"] is True
        assert details["pii_leakage"] is False  # never reached
        mock_judge.assert_not_called()


def test_pii_redaction_and_toxicity_block_can_coexist_in_details():
    """PII redact-and-continue still records pii_leakage even when the
    NEXT check (toxicity) is what ultimately blocks the response."""
    with patch("core.output_guardrails.output_judge", return_value="DANGEROUS"):
        decision, details = assess_output("Contact john.doe@example.com, this is harmful.")
        assert decision == "BLOCK"
        assert details["pii_leakage"] is True
        assert details["toxicity_detected"] is True
        assert details["clean_response"] is None


def test_no_system_prompt_leak_check_skips_even_with_long_response():
    """Without a reference system_prompt, no amount of response length or
    repetition should trigger a leak finding."""
    with patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        decision, details = assess_output("x" * 500, system_prompt=None)
        assert decision == "ALLOW"
        assert details["system_prompt_leak_detected"] is False


def test_assess_output_details_shape_is_stable_on_clean_pass():
    with patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        decision, details = assess_output("a perfectly ordinary response")
        assert decision == "ALLOW"
        assert details["source"] == "clean_pass"
        assert set(details.keys()) == {
            "secrets_detected", "system_prompt_leak_detected", "pii_leakage",
            "toxicity_detected", "hallucination_detected", "source", "clean_response",
        }


def test_assess_output_empty_response_text_does_not_crash():
    with patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        decision, details = assess_output("")
        assert decision == "ALLOW"
        assert details["clean_response"] == ""


# ---------------------------------------------------------------------------
# check_semantic_grounding -- real cosine_similarity math (no mocked score)
# ---------------------------------------------------------------------------

def test_grounding_real_cosine_math_identical_vectors_passes():
    pytest.importorskip("sentence_transformers")  # real cosine_similarity() needs it
    with patch("core.output_guardrails.educational_store") as store, \
         patch("core.output_guardrails.get_embedding", return_value=[1.0, 0.0, 0.0]):
        store.get_centroid.return_value = [1.0, 0.0, 0.0]
        assert check_semantic_grounding("aligned") is True


def test_grounding_real_cosine_math_orthogonal_vectors_fails():
    pytest.importorskip("sentence_transformers")  # real cosine_similarity() needs it
    with patch("core.output_guardrails.educational_store") as store, \
         patch("core.output_guardrails.get_embedding", return_value=[1.0, 0.0, 0.0]):
        store.get_centroid.return_value = [0.0, 1.0, 0.0]
        assert check_semantic_grounding("orthogonal") is False


def test_leakage_finds_run_appearing_multiple_times_in_response():
    run = "z" * 45
    system_prompt = f"AAAAA{run}BBBBB"
    response = f"{run} then later again {run}"
    assert check_system_prompt_leakage(response, system_prompt, min_run=40) is True


def test_assess_output_toxicity_safe_verdict_string_is_case_sensitive():
    """Only the exact 'DANGEROUS' string blocks -- anything else, including
    a differently-cased variant, is treated as SAFE by the equality check."""
    with patch("core.output_guardrails.output_judge", return_value="dangerous"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        decision, details = assess_output("some response")
        assert decision == "ALLOW"
        assert details["toxicity_detected"] is False
