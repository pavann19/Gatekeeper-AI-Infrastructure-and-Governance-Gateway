# core/output_guardrails.py
from core.config import settings
from core.privacy import redact_pii
from core.secrets_detection import detect_secrets
from core.semantic_judge import output_judge
from core.logger import get_logger
from core.embeddings import get_embedding, cosine_similarity
from core.vector_store import educational_store

logger = get_logger(__name__)

# Minimum contiguous run of the caller-supplied system prompt that must
# appear verbatim in the response to count as a leak. Short shared runs
# ("you are a helpful assistant") are common, generic phrasing that would
# make this check noisy at a low threshold; a run this long surviving
# verbatim is a strong, specific signal of the actual prompt text leaking,
# not incidental phrasing overlap.
SYSTEM_PROMPT_LEAK_MIN_RUN = 40


def check_semantic_grounding(response_text: str) -> bool:
    """
    Checks if the response is semantically grounded in our acceptable domains.
    Returns False if it severely diverges (hallucination).
    """
    centroid = educational_store.get_centroid()
    if centroid is None:
        return True # Not enough data to block

    vec = get_embedding(response_text)
    score = cosine_similarity(vec, centroid)

    # If the response is almost completely orthogonal to the domain centroid, it's flagged
    if score < 0.15:
        return False
    return True


def check_system_prompt_leakage(response_text: str, system_prompt: str,
                                min_run: int = SYSTEM_PROMPT_LEAK_MIN_RUN) -> bool:
    """
    True if a contiguous run of at least `min_run` characters from
    `system_prompt` appears verbatim in `response_text`.

    Deliberately a verbatim substring check, not a semantic-similarity one:
    "leaked its system prompt" means the LLM reproduced the actual text, not
    that its response is topically similar to it — a similarity score would
    conflate "this response is ABOUT the same subject as the system prompt"
    (expected, harmless) with "this response CONTAINS the system prompt"
    (the actual incident). Sliding-window substring matching answers the
    second question directly.
    """
    if not system_prompt or len(system_prompt) < min_run:
        return False
    for i in range(len(system_prompt) - min_run + 1):
        if system_prompt[i:i + min_run] in response_text:
            return True
    return False


def assess_output(response_text: str, system_prompt: str = None) -> tuple:
    """
    Evaluates the LLM's response before it is returned to the user.

    Checks, in order (cheapest/most severe first, mirroring core/risk.py's
    own hard-veto-before-expensive-checks shape):
    1. Secrets (API keys, tokens, private key blocks) — BLOCK. No safe
       partial version of a leaked credential exists, so there is nothing to
       redact-and-continue here, unlike PII below.
    2. System-prompt leakage (only if `system_prompt` was supplied) — BLOCK.
    3. PII — REDACT AND CONTINUE, not block (Phase 2, Output Security). The
       previous behaviour blocked the entire response over a single
       incidental PII match, discarding the redaction it had already
       computed. This mirrors the INPUT side's own behaviour
       (core/risk.py's collect_semantic_signals runs on the PII-redacted
       prompt, never on the raw one) rather than treating input and output
       PII inconsistently.
    4. Toxicity / harmful content, via Semantic Judge — BLOCK. Runs on the
       PII-redacted text, not the raw text, for the same reason (2) redacts
       rather than reveals: a toxicity judge does not need to see PII to do
       its job, so there is no reason to widen exposure of it further.
    5. Semantic Grounding Check (hallucination) — BLOCK. Same redacted text.

    Returns:
        (decision, details) where decision is "ALLOW" or "BLOCK".
        details["clean_response"] carries the PII-redacted text whenever the
        response was not blocked outright — None if it was.
    """
    details = {
        "secrets_detected": False,
        "system_prompt_leak_detected": False,
        "pii_leakage": False,
        "toxicity_detected": False,
        "hallucination_detected": False,
        "source": "clean_pass",
        "clean_response": None,
    }

    # 1. Secret detection — hard block, no redact-and-continue (see docstring).
    secret_result = detect_secrets(response_text)
    if secret_result["secrets_found"]:
        logger.warning(f"🚫 OUTPUT BLOCKED: Secret(s) detected in response: {secret_result['items']}")
        details["secrets_detected"] = True
        details["source"] = "secret_leakage"
        details["secret_items"] = secret_result["items"]
        return "BLOCK", details

    # 2. System-prompt leakage — only evaluated when the caller supplied a
    #    reference prompt to check against (see AssessOutputRequest.system_prompt).
    if system_prompt and check_system_prompt_leakage(response_text, system_prompt):
        logger.warning("🚫 OUTPUT BLOCKED: Response leaked the supplied system prompt verbatim.")
        details["system_prompt_leak_detected"] = True
        details["source"] = "system_prompt_leakage"
        return "BLOCK", details

    # 3. PII — redact and continue.
    clean_text, redacted_info = redact_pii(response_text)
    if redacted_info.get("pii_found"):
        logger.info(f"Output PII redacted (not blocked). Source: {redacted_info.get('source')}")
        details["pii_leakage"] = True
        details["redacted_items"] = redacted_info.get("items")

    # 4. Toxicity Check via Local LLM Judge — on the redacted text.
    verdict = output_judge(clean_text)
    if verdict == "DANGEROUS":
        logger.warning("🚫 OUTPUT BLOCKED: Semantic Judge classified response as TOXIC/HARMFUL.")
        details["toxicity_detected"] = True
        details["source"] = "semantic_judge_output"
        return "BLOCK", details

    # 5. Semantic Grounding Check (Hallucination) — on the redacted text.
    #    Gated by HALLUCINATION_CHECK_ENABLED (default False): see that
    #    setting's docstring in core/config.py for why -- this check
    #    currently borrows an INPUT-side anchor set as its reference
    #    "expected domain", a category error a live pentest caught blocking
    #    ordinary, legitimate output. Left wired up, not deleted, so it can
    #    be re-enabled once a real output-domain corpus replaces the
    #    borrowed one.
    if settings.HALLUCINATION_CHECK_ENABLED and not check_semantic_grounding(clean_text):
        logger.warning("🚫 OUTPUT BLOCKED: Response diverged significantly from expected semantic domain (Hallucination).")
        details["hallucination_detected"] = True
        details["source"] = "semantic_grounding_check"
        return "BLOCK", details

    if details["pii_leakage"]:
        details["source"] = "pii_redacted_pass"
    details["clean_response"] = clean_text
    return "ALLOW", details
