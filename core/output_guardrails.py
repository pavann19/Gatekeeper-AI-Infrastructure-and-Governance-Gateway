# core/output_guardrails.py
from core.privacy import redact_pii
from core.semantic_judge import output_judge
from core.logger import get_logger
from core.embeddings import get_embedding, cosine_similarity
from core.vector_store import educational_store
import asyncio

logger = get_logger(__name__)

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

def assess_output(response_text: str) -> tuple:
    """
    Evaluates the LLM's response before it is returned to the user.
    Checks for:
    1. PII Leakage (Did the LLM output raw PII?)
    2. Toxicity / Harmful content (via Semantic Judge)
    
    Returns:
        (decision, details) where decision is "ALLOW" or "BLOCK"
    """
    details = {
        "pii_leakage": False,
        "toxicity_detected": False,
        "hallucination_detected": False,
        "source": "clean_pass"
    }
    
    # 1. PII Leakage Check
    clean_text, redacted_info = redact_pii(response_text)
    if redacted_info.get("pii_found"):
        logger.warning(f"⚠️ OUTPUT BLOCKED: PII Leakage Detected. Source: {redacted_info.get('source')}")
        details["pii_leakage"] = True
        details["source"] = "pii_leakage"
        details["redacted_items"] = redacted_info.get("items")
        return "BLOCK", details
        
    # 2. Toxicity Check via Local LLM Judge
    verdict = output_judge(response_text)
    if verdict == "DANGEROUS":
        logger.warning("🚫 OUTPUT BLOCKED: Semantic Judge classified response as TOXIC/HARMFUL.")
        details["toxicity_detected"] = True
        details["source"] = "semantic_judge_output"
        return "BLOCK", details
        
    # 3. Semantic Grounding Check (Hallucination)
    if not check_semantic_grounding(response_text):
        logger.warning("🚫 OUTPUT BLOCKED: Response diverged significantly from expected semantic domain (Hallucination).")
        details["hallucination_detected"] = True
        details["source"] = "semantic_grounding_check"
        return "BLOCK", details
        
    return "ALLOW", details
