"""
Edge-case coverage for core/secrets_detection.py, complementing
tests/test_secrets_detection.py (which covers one happy-path example per
pattern plus the capturing-group regression guard).

This file focuses on:
  - every pattern label detected with an exact, expected type
  - realistic near-miss text that must NOT be flagged (false-positive
    resistance)
  - multiple distinct secrets in one text, all detected -- not just the
    first (a real risk given re.findall() is called once per pattern in
    a loop and results are appended into a single flat list)
  - secrets embedded mid-sentence vs. standalone
  - case sensitivity of each pattern
"""
from core.secrets_detection import detect_secrets


def _labels(result):
    return [item.split(":", 1)[0] for item in result["items"]]


# ---------------------------------------------------------------------
# One exact-type assertion per pattern the module claims to detect
# ---------------------------------------------------------------------

def test_asia_prefixed_aws_key_detected_with_exact_type():
    result = detect_secrets("temp creds ASIAABCDEFGHIJKLMNOP in use")
    assert result["secrets_found"] is True
    assert _labels(result) == ["AWS_ACCESS_KEY"]


def test_github_oauth_token_prefix_gho_detected():
    result = detect_secrets("export TOKEN=gho_" + "B" * 36)
    assert result["secrets_found"] is True
    assert _labels(result) == ["GITHUB_TOKEN"]


def test_github_user_token_prefix_ghu_detected():
    result = detect_secrets("ghu_" + "c" * 40)
    assert result["secrets_found"] is True
    assert _labels(result) == ["GITHUB_TOKEN"]


def test_slack_app_token_prefix_xoxa_detected():
    result = detect_secrets("xoxa-2-FAKE-TEST-TOKEN-000000-not-real")
    assert result["secrets_found"] is True
    assert _labels(result) == ["SLACK_TOKEN"]


def test_google_api_key_detected_with_exact_type():
    fake_google_key = "AIza" + "S" * 35
    result = detect_secrets(f"key={fake_google_key}")
    assert result["secrets_found"] is True
    assert _labels(result) == ["GOOGLE_API_KEY"]


def test_anthropic_key_detected_as_anthropic_only_not_openai():
    """
    OPENAI_API_KEY's pattern requires 20+ [A-Za-z0-9] chars directly after
    'sk-'. An anthropic key's actual next characters are 'ant-' -- the
    hyphen breaks the [A-Za-z0-9]{20,} run after only 3 chars ("ant"),
    so the OPENAI pattern does NOT match here even though it looks like
    a textual superset at a glance. Only ANTHROPIC_API_KEY's own
    (sk-ant-[A-Za-z0-9-]{20,}) pattern, which allows hyphens, matches.
    This locks in that actual (not assumed) behavior.
    """
    fake_anthropic_key = "sk-ant-" + "a" * 24
    result = detect_secrets(fake_anthropic_key)
    assert result["secrets_found"] is True
    assert _labels(result) == ["ANTHROPIC_API_KEY"]


def test_openssh_private_key_header_detected():
    result = detect_secrets("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1r...")
    assert result["secrets_found"] is True
    assert _labels(result) == ["PRIVATE_KEY_BLOCK"]


def test_dsa_private_key_header_detected():
    result = detect_secrets("-----BEGIN DSA PRIVATE KEY-----\nMIIBuw...")
    assert result["secrets_found"] is True
    assert _labels(result) == ["PRIVATE_KEY_BLOCK"]


def test_generic_unqualified_private_key_header_detected():
    result = detect_secrets("-----BEGIN PRIVATE KEY-----\nMIIEvQ...")
    assert result["secrets_found"] is True
    assert _labels(result) == ["PRIVATE_KEY_BLOCK"]


# ---------------------------------------------------------------------
# False-positive resistance: realistic near-misses that must NOT match
# ---------------------------------------------------------------------

def test_short_sk_prefixed_word_is_not_flagged_as_openai_key():
    """'sk-' alone doesn't meet the 20+ char requirement -- ordinary text
    using that prefix (e.g. a SKU code) must not be flagged."""
    result = detect_secrets("Item code sk-42 is out of stock.")
    assert result["secrets_found"] is False


def test_low_entropy_repeated_long_string_is_not_flagged():
    """Module explicitly does NOT do generic high-entropy scanning, so a
    long but low-entropy repeated string must never be flagged, even
    though it is superficially 'long and random-looking'."""
    result = detect_secrets("Padding: " + "ab" * 40)
    assert result["secrets_found"] is False


def test_aws_like_prefix_with_wrong_length_not_flagged():
    """AKIA followed by too few characters should not match the 16-char
    suffix requirement."""
    result = detect_secrets("AKIASHORT123")
    assert result["secrets_found"] is False


def test_github_like_prefix_with_short_suffix_not_flagged():
    result = detect_secrets("ghp_tooshort")
    assert result["secrets_found"] is False


def test_uuid_request_id_mid_sentence_not_flagged():
    result = detect_secrets(
        "The failed request had id 6f9619ff-8b86-d011-b42d-00cf4fc964ff "
        "and was retried twice."
    )
    assert result["secrets_found"] is False


def test_generic_bearer_word_without_shaped_token_not_flagged():
    result = detect_secrets("Please include a Bearer token in the Authorization header.")
    assert result["secrets_found"] is False


# ---------------------------------------------------------------------
# Multiple distinct secrets in the same text -- all must be detected
# ---------------------------------------------------------------------

def test_multiple_distinct_secret_types_all_detected():
    aws_key = "AKIA" + "Q" * 16
    github_token = "ghp_" + "d" * 36
    text = (
        f"Rotate these immediately: aws key {aws_key} and github token "
        f"{github_token} were committed by mistake."
    )
    result = detect_secrets(text)
    assert result["secrets_found"] is True
    labels = set(_labels(result))
    assert labels == {"AWS_ACCESS_KEY", "GITHUB_TOKEN"}
    assert len(result["items"]) == 2


def test_two_occurrences_of_same_pattern_both_detected():
    key1 = "AKIA" + "A" * 16
    key2 = "AKIA" + "B" * 16
    result = detect_secrets(f"old key {key1} replaced by new key {key2}")
    assert result["secrets_found"] is True
    assert _labels(result) == ["AWS_ACCESS_KEY", "AWS_ACCESS_KEY"]
    assert result["items"][0] != result["items"][1]


def test_three_distinct_secret_types_including_private_key_block():
    slack_token = "xoxb-" + "1" * 12
    google_key = "AIza" + "Z" * 35
    text = (
        "Leaked in the log dump:\n"
        f"{slack_token}\n"
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpQIBAAKCAQEA\n"
        f"and also {google_key} in the query string."
    )
    result = detect_secrets(text)
    labels = set(_labels(result))
    assert labels == {"SLACK_TOKEN", "PRIVATE_KEY_BLOCK", "GOOGLE_API_KEY"}
    assert len(result["items"]) == 3


# ---------------------------------------------------------------------
# Mid-sentence vs. standalone placement
# ---------------------------------------------------------------------

def test_aws_key_standalone_is_detected():
    result = detect_secrets("AKIA" + "X" * 16)
    assert result["secrets_found"] is True
    assert _labels(result) == ["AWS_ACCESS_KEY"]


def test_aws_key_embedded_mid_sentence_is_detected():
    key = "AKIA" + "X" * 16
    result = detect_secrets(
        f"By the way, the deploy script still hardcodes {key} in a comment, "
        "which someone should really fix before the next release."
    )
    assert result["secrets_found"] is True
    assert _labels(result) == ["AWS_ACCESS_KEY"]


def test_jwt_embedded_mid_sentence_is_detected():
    fake_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoidGVzdCJ9.c2lnbmF0dXJlLXBhcnQ"
    result = detect_secrets(
        f"When I inspected the request headers I noticed {fake_jwt} sitting "
        "right there in plaintext, which seems wrong."
    )
    assert result["secrets_found"] is True
    assert _labels(result) == ["JWT"]


# ---------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------

def test_lowercase_akia_prefix_not_detected():
    """Patterns are not compiled with re.IGNORECASE, so a lowercased
    prefix must not match -- locking in current case-sensitive behavior."""
    result = detect_secrets("akia" + "x" * 16)
    assert result["secrets_found"] is False


def test_uppercase_github_prefix_not_detected():
    result = detect_secrets("GHP_" + "A" * 36)
    assert result["secrets_found"] is False


def test_mixed_case_slack_prefix_not_detected():
    result = detect_secrets("Xoxb-" + "a" * 12)
    assert result["secrets_found"] is False


def test_private_key_header_wrong_case_not_detected():
    result = detect_secrets("-----begin rsa private key-----\nabc")
    assert result["secrets_found"] is False


def test_aws_key_suffix_lowercase_not_detected():
    """The AKIA prefix is correct case, but the required [0-9A-Z]{16}
    suffix must be uppercase -- a lowercase suffix must not match."""
    result = detect_secrets("AKIA" + "a" * 16)
    assert result["secrets_found"] is False
