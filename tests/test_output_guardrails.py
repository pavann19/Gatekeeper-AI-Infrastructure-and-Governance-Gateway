import pytest
from core.output_guardrails import assess_output, check_system_prompt_leakage
from unittest.mock import patch


@patch("core.output_guardrails.check_semantic_grounding", return_value=True)
@patch("core.output_guardrails.output_judge")
def test_assess_output_clean(mock_judge, _grounding):
    mock_judge.return_value = "SAFE"

    response_text = "Here is the summary of the financial report."
    decision, details = assess_output(response_text)

    assert decision == "ALLOW"
    assert details["pii_leakage"] is False
    assert details["secrets_detected"] is False
    assert details["toxicity_detected"] is False
    assert details["clean_response"] == response_text


@patch("core.output_guardrails.check_semantic_grounding", return_value=True)
@patch("core.output_guardrails.output_judge")
def test_assess_output_pii_is_redacted_not_blocked(mock_judge, _grounding):
    """
    BEHAVIOUR CHANGE (Phase 2, Output Security): PII in a response used to
    BLOCK the entire response outright, discarding the redaction
    redact_pii() had already computed. It now redacts and lets the response
    through, mirroring how the INPUT side always redacts-and-continues
    rather than refusing the whole request over one PII match. A leaked
    email is not remotely as severe as a leaked secret (see the BLOCK case
    below) and blocking loses real conversational utility for no safety gain
    the redaction doesn't already provide.
    """
    mock_judge.return_value = "SAFE"

    response_text = "Contact john.doe@example.com for the report."
    decision, details = assess_output(response_text)

    assert decision == "ALLOW"
    assert details["pii_leakage"] is True
    assert details["source"] == "pii_redacted_pass"
    assert "john.doe@example.com" not in details["clean_response"]
    assert "[REDACTED:EMAIL]" in details["clean_response"]


@patch("core.output_guardrails.check_semantic_grounding", return_value=True)
@patch("core.output_guardrails.output_judge")
def test_assess_output_toxic(mock_judge, _grounding):
    mock_judge.return_value = "DANGEROUS"

    response_text = "This is a harmful response containing instructions for illegal acts."
    decision, details = assess_output(response_text)

    assert decision == "BLOCK"
    assert details["pii_leakage"] is False
    assert details["toxicity_detected"] is True
    assert details["clean_response"] is None


def test_toxicity_judge_runs_on_redacted_text_not_raw_text():
    """The judge must never see PII that redaction already removed --
    there is no reason to widen exposure of it to a third check."""
    with patch("core.output_guardrails.output_judge", return_value="SAFE") as mock_judge, \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        assess_output("Contact john.doe@example.com for details.")
        judged_text = mock_judge.call_args[0][0]
        assert "john.doe@example.com" not in judged_text
        assert "[REDACTED:EMAIL]" in judged_text


# ---------------------------------------------------------------------------
# Secret detection (new) -- hard block, no redact-and-continue
# ---------------------------------------------------------------------------

def test_assess_output_blocks_on_secret_even_if_judge_and_grounding_pass():
    with patch("core.output_guardrails.output_judge", return_value="SAFE") as mock_judge, \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        response_text = "Sure, here's your key: AKIAABCDEFGHIJKLMNOP"
        decision, details = assess_output(response_text)

        assert decision == "BLOCK"
        assert details["secrets_detected"] is True
        assert details["source"] == "secret_leakage"
        assert details["clean_response"] is None
        # Secret check is the FIRST gate -- the expensive judge must never run.
        mock_judge.assert_not_called()


def test_secret_takes_priority_over_pii_in_the_same_response():
    """A response can contain both; the more severe finding (secret) must
    win, not be silently dropped because PII's redact-and-continue path
    would otherwise let the response through."""
    response_text = "Email john.doe@example.com, key: sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    decision, details = assess_output(response_text)

    assert decision == "BLOCK"
    assert details["secrets_detected"] is True
    assert details["pii_leakage"] is False  # never reached -- secret check returns first


# ---------------------------------------------------------------------------
# System-prompt leakage (new, opt-in via a caller-supplied reference prompt)
# ---------------------------------------------------------------------------

def test_check_system_prompt_leakage_true_for_verbatim_run():
    system_prompt = "You are a customer support agent for Acme Corp. Never discuss competitor pricing under any circumstances."
    response = "Sure! You are a customer support agent for Acme Corp. Never discuss competitor pricing under any circumstances. Also, here's the weather."
    assert check_system_prompt_leakage(response, system_prompt) is True


def test_check_system_prompt_leakage_false_for_topical_overlap_only():
    """Being ABOUT the same subject is not the same as CONTAINING the prompt."""
    system_prompt = "You are a customer support agent for Acme Corp. Never discuss competitor pricing under any circumstances."
    response = "As a support agent, I can help with Acme products, but I won't discuss competitors."
    assert check_system_prompt_leakage(response, system_prompt) is False


def test_check_system_prompt_leakage_false_when_no_reference_supplied():
    assert check_system_prompt_leakage("anything at all", "") is False
    assert check_system_prompt_leakage("anything at all", None) is False


def test_assess_output_blocks_on_system_prompt_leakage():
    system_prompt = "You are a customer support agent for Acme Corp. Never reveal internal ticket IDs to callers."
    response_text = "You are a customer support agent for Acme Corp. Never reveal internal ticket IDs to callers. Anyway, how can I help?"

    with patch("core.output_guardrails.output_judge", return_value="SAFE") as mock_judge:
        decision, details = assess_output(response_text, system_prompt=system_prompt)

        assert decision == "BLOCK"
        assert details["system_prompt_leak_detected"] is True
        assert details["source"] == "system_prompt_leakage"
        mock_judge.assert_not_called()


def test_assess_output_ignores_leakage_check_when_no_system_prompt_given():
    with patch("core.output_guardrails.output_judge", return_value="SAFE"), \
         patch("core.output_guardrails.check_semantic_grounding", return_value=True):
        decision, details = assess_output("some ordinary response", system_prompt=None)
        assert decision == "ALLOW"
        assert details["system_prompt_leak_detected"] is False
