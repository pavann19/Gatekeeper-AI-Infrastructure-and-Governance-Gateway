# core/secrets_detection.py
"""Detect credential-like secrets in untrusted input and generated output."""
import re

SECRET_PATTERNS = {
    "GATEKEEPER_API_KEY": r"(?<![A-Za-z0-9_-])gk_[A-Za-z0-9_-]{43,}(?![A-Za-z0-9_-])",
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
