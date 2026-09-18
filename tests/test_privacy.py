import pytest
from core.privacy import redact_pii

def test_redact_pii_email():
    prompt = "My email is test@example.com."
    clean, metadata = redact_pii(prompt)
    assert "test@example.com" not in clean
    assert "[REDACTED:EMAIL]" in clean
    assert metadata["pii_found"] is True

def test_redact_pii_phone():
    prompt = "Call me at +1 555 123 4567."
    clean, metadata = redact_pii(prompt)
    assert "+1 555 123 4567" not in clean
    assert "[REDACTED:PHONE]" in clean

def test_redact_pii_short_local_phone():
    # Fictional 7-digit format (no area code), e.g. movie/demo phone numbers.
    prompt = "Call John Doe at 555-0199 and check system health."
    clean, metadata = redact_pii(prompt)
    assert "555-0199" not in clean
    assert "[REDACTED:PHONE]" in clean
    assert "[REDACTED:PERSON]" in clean
    assert "PHONE:555-0199" in metadata["items"]
    assert "PERSON:John Doe" in metadata["items"]

def test_redact_pii_aadhaar():
    prompt = "My ID is 1234 5678 9012"
    clean, metadata = redact_pii(prompt)
    assert "1234 5678 9012" not in clean
    assert "[REDACTED:AADHAAR]" in clean

def test_redact_pii_clean():
    prompt = "Hello world, what is the capital?"
    clean, metadata = redact_pii(prompt)
    assert clean == prompt
    assert metadata.get("source") in ["CLEAN", "NER_CONTEXT", "NONE"]
