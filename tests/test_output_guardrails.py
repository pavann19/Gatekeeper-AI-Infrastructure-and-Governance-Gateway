import pytest
from core.output_guardrails import assess_output
from unittest.mock import patch

@patch("core.output_guardrails.output_judge")
def test_assess_output_clean(mock_judge):
    # Mock judge to return SAFE
    mock_judge.return_value = "SAFE"
    
    response_text = "Here is the summary of the financial report."
    decision, details = assess_output(response_text)
    
    assert decision == "ALLOW"
    assert details["pii_leakage"] is False
    assert details["toxicity_detected"] is False

@patch("core.output_guardrails.output_judge")
def test_assess_output_pii_leak(mock_judge):
    # Even if judge says SAFE, PII should trigger BLOCK
    mock_judge.return_value = "SAFE"
    
    response_text = "The password for john.doe@example.com has been reset."
    decision, details = assess_output(response_text)
    
    assert decision == "BLOCK"
    assert details["pii_leakage"] is True
    assert details["toxicity_detected"] is False

@patch("core.output_guardrails.output_judge")
def test_assess_output_toxic(mock_judge):
    # Mock judge to return DANGEROUS
    mock_judge.return_value = "DANGEROUS"
    
    response_text = "This is a harmful response containing instructions for illegal acts."
    decision, details = assess_output(response_text)
    
    assert decision == "BLOCK"
    assert details["pii_leakage"] is False
    assert details["toxicity_detected"] is True
