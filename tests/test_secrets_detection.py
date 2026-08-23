from core.secrets_detection import detect_secrets


def test_detects_aws_access_key():
    result = detect_secrets("Here is your key: AKIAABCDEFGHIJKLMNOP")
    assert result["secrets_found"] is True
    assert any(item.startswith("AWS_ACCESS_KEY:") for item in result["items"])


def test_detects_github_token():
    result = detect_secrets("token: ghp_" + "a" * 36)
    assert result["secrets_found"] is True
    assert any(item.startswith("GITHUB_TOKEN:") for item in result["items"])


def test_detects_slack_token():
    result = detect_secrets("xoxb-FAKE-TEST-TOKEN-NOT-REAL-000000")
    assert result["secrets_found"] is True
    assert any(item.startswith("SLACK_TOKEN:") for item in result["items"])


def test_detects_openai_style_key():
    result = detect_secrets("sk-" + "a" * 24)
    assert result["secrets_found"] is True
    assert any(item.startswith("OPENAI_API_KEY:") for item in result["items"])


def test_detects_jwt():
    fake_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQ_dGVzdHNpZw"
    result = detect_secrets(f"Your token is {fake_jwt}")
    assert result["secrets_found"] is True
    assert any(item.startswith("JWT:") for item in result["items"])


def test_detects_private_key_block():
    result = detect_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...")
    assert result["secrets_found"] is True
    assert any(item.startswith("PRIVATE_KEY_BLOCK:") for item in result["items"])


def test_clean_text_reports_no_secrets():
    result = detect_secrets("The weather today is sunny with a high of 75 degrees.")
    assert result["secrets_found"] is False
    assert result["items"] == []


def test_ordinary_hash_or_uuid_is_not_flagged():
    """Guards the false-positive discipline the module docstring commits
    to: a generic random-looking string must NOT match any pattern here,
    only vendor-specific prefixed formats."""
    result = detect_secrets("Request ID: 550e8400-e29b-41d4-a716-446655440000")
    assert result["secrets_found"] is False


def test_matches_are_never_returned_in_full():
    key = "AKIAABCDEFGHIJKLMNOP"
    result = detect_secrets(key)
    for item in result["items"]:
        assert key not in item
        assert "..." in item


def test_no_pattern_has_a_capturing_group():
    """
    re.findall() silently returns ONLY a capturing group's content when a
    pattern has one, not the full match -- exactly the bug the AWS and
    private-key patterns both had (findall returned "AKIA", truncating the
    key before the preview slice even ran). Guards every pattern in the
    module against reintroducing it.
    """
    import re as re_module
    from core.secrets_detection import SECRET_PATTERNS
    for label, pattern in SECRET_PATTERNS.items():
        compiled = re_module.compile(pattern)
        assert compiled.groups == 0, (
            f"{label}'s pattern has a capturing group -- use (?:...) instead, "
            f"or re.findall() will return only the group's content."
        )
