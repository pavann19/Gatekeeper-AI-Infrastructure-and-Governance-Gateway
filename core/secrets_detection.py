# core/secrets_detection.py
"""
Detects credential-shaped strings (API keys, tokens, private key blocks) in
LLM-generated output. Phase 2 (Output Security) roadmap item.

WHY THIS IS SEPARATE FROM core/privacy.py's PII REDACTION
-----------------------------------------------------------
PII (an email, a name) is often incidental and conversationally useful --
redacting it and letting the rest of the response through preserves utility.
A live credential is categorically different: there is no "mostly safe"
version of a response that contains a working API key. Detection here
therefore reports a hard signal the caller (core/output_guardrails.py)
treats as an unconditional BLOCK, not a redact-and-continue candidate.

WHY REGEX, matching core/privacy.py's REGEX_PATTERNS approach: every pattern
below matches a vendor-specific, structurally distinctive prefix (AKIA...,
sk-..., ghp_..., xox?-..., a PEM header). These are the same class of
high-precision, low-recall patterns public secret-scanners (gitleaks,
trufflehog) lead with, chosen specifically to avoid the false-positive cost
of a generic "looks like a random string" heuristic, which would flag
ordinary hashes, UUIDs, and base64 data as "secrets" and make this
unusable. A generic high-entropy scan is explicitly NOT attempted here for
that reason -- narrower coverage, but every hit is a real, actionable
finding rather than noise a deployment would learn to ignore.
"""
import re

SECRET_PATTERNS = {
    # Non-capturing group deliberately: re.findall() returns ONLY a
    # capturing group's content when a pattern has one, not the full match
    # -- a real bug this module's own tests caught (findall returned just
    # "AKIA", truncating the key before the preview slice even ran).
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "SLACK_TOKEN": r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b",
    "OPENAI_API_KEY": r"\bsk-[A-Za-z0-9]{20,}\b",
    "ANTHROPIC_API_KEY": r"\bsk-ant-[A-Za-z0-9-]{20,}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z\-_]{35}\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
    "PRIVATE_KEY_BLOCK": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
}


def detect_secrets(text: str) -> dict:
    """
    Scans text for credential-shaped strings.

    Returns:
        {"secrets_found": bool, "items": [f"{LABEL}:{redacted_preview}", ...]}

    Matches are never returned in full — only a short, non-reconstructible
    preview (first 6 chars + "...") — so the audit trail this feeds
    (core/logger.py's log_output_event) documents THAT a secret leaked and
    of what kind, without itself becoming a second place the secret is
    stored in full.
    """
    items = []
    for label, pattern in SECRET_PATTERNS.items():
        for match in re.findall(pattern, text):
            preview = match[:6] + "..." if len(match) > 6 else match
            items.append(f"{label}:{preview}")

    return {"secrets_found": bool(items), "items": items}
