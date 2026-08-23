# core/logger.py
import hashlib
import os
from datetime import datetime
import logging
from pythonjsonlogger import jsonlogger

from core.config import settings

# --- CONFIGURE STRUCTURED LOGGING ---
logger = logging.getLogger("gatekeeper")
logger.setLevel(logging.INFO)

# Configurable so a container deployment can point this at a mounted volume
# — "audit.jsonl" (relative to cwd) previously meant every container
# recreation silently discarded the ENTIRE audit trail, which is a real
# compliance defect for a project whose audit record is repeatedly
# documented elsewhere as the compliance artefact. Default is unchanged
# from before this setting existed, so nothing that already works differs.
AUDIT_LOG_PATH = settings.AUDIT_LOG_PATH

if not logger.handlers:
    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # JSON Audit Handler
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH) or ".", exist_ok=True)
    audit_handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
    json_formatter = jsonlogger.JsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    audit_handler.setFormatter(json_formatter)
    
    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False 
    audit_logger.addHandler(audit_handler)

def get_logger(name="gatekeeper"):
    return logging.getLogger(name)

def log_event(capability, prompt, risk, decision, metadata=None):
    if metadata is None:
        metadata = {}
    
    timestamp = datetime.now().isoformat()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    log_entry = {
        "timestamp": timestamp,
        # Distinguishes this record from log_output_event's records in the
        # same audit stream — see that function's docstring (Phase 2, Output
        # Security) for why they are separate events rather than one shape
        # with an optional half.
        "event_type": "input_assessment",
        # Correlation ID, propagated from ingress (api/main.py) so an audit
        # record can be tied back to the HTTP request, its access-log line and
        # any downstream call that carried the same header. Without it, the
        # only join key between a governance decision and the request that
        # caused it is a timestamp, which stops being unique under any real
        # concurrency.
        "request_id": metadata.get("request_id", "unset"),
        # Top-level, not just nested in "principal" (which also carries it) —
        # "show every decision for tenant X" is a query an auditor actually
        # runs, and it should not require reaching into a nested object to
        # answer. Named for the Tenant Resolver work (core/tenancy.py); a
        # record from before that landed reads "unset", not "default", so a
        # query can distinguish "no tenant concept existed yet" from "this
        # caller resolved to the default tenant".
        "tenant": metadata.get("tenant", "unset"),
        "capability": capability,
        "risk": risk,
        "decision": decision,
        "prompt_hash": prompt_hash,
        "semantic_score": metadata.get("semantic_score", 0.0),
        "source": metadata.get("source", "unknown"),
        "educational_context": metadata.get("educational_context", False),
        "domain_score": metadata.get("domain_score", None),
        "symbolic_triggered": metadata.get("symbolic_triggered", False),
        "judge_invoked": metadata.get("judge_invoked", False),
    }

    # Structured JSON log for Audit
    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.info("Governance Decision", extra=log_entry)


def log_output_event(capability, response_text, decision, metadata=None, tenant="unset", request_id="unset"):
    """
    Audit record for an OUTPUT assessment — distinct from log_event's INPUT
    record (Phase 2, Output Security roadmap item).

    WHY A SEPARATE FUNCTION, not log_event with more optional fields. An
    input assessment and an output assessment answer different questions
    ("is this prompt an attack?" vs "is this response safe to return?") and
    carry fields that don't apply to the other (risk/symbolic_triggered vs
    pii_leakage/secrets_detected/system_prompt_leak). Cramming both into one
    schema with everything optional makes every consumer of the audit log —
    a query, a dashboard, a compliance export — responsible for knowing
    which half of the record is meaningful for a given row. `event_type`
    lets a query select cleanly; the two record shapes stay honest about
    what they actually measure.

    Previously MISSING ENTIRELY for the standalone /api/v1/assess_output
    endpoint — that endpoint could BLOCK a response (PII leak, toxicity,
    secret, hallucination) with zero audit trail. Every output decision must
    be as auditable as every input decision; this closes that gap.
    """
    if metadata is None:
        metadata = {}

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "output_assessment",
        "request_id": request_id,
        "tenant": tenant,
        "capability": capability,
        "decision": decision,
        "response_hash": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "pii_leakage": metadata.get("pii_leakage", False),
        "secrets_detected": metadata.get("secrets_detected", False),
        "toxicity_detected": metadata.get("toxicity_detected", False),
        "hallucination_detected": metadata.get("hallucination_detected", False),
        "system_prompt_leak_detected": metadata.get("system_prompt_leak_detected", False),
        "source": metadata.get("source", "unknown"),
    }

    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.info("Output Governance Decision", extra=log_entry)


def log_gateway_event(capability, provider, model, success, latency_ms, decision,
                      usage=None, error=None, tenant="unset", request_id="unset"):
    """
    Audit record for a PROXIED CALL through the LLM gateway (Phase 5: Real
    LLM Gateway, "Audit trail for the proxied call itself" roadmap item) —
    a THIRD distinct event_type alongside log_event's "input_assessment"
    and log_output_event's "output_assessment", for the same reason those
    two are separate from each other: this answers a third, different
    question ("did the call to the external provider succeed, and what did
    it cost?") that neither the input nor output assessment records answer.

    Deliberately does not duplicate the input/output governance verdicts —
    those are already fully recorded by log_event/log_output_event for the
    same request_id, joinable on it. `decision` here is the FINAL combined
    decision the gateway returned to the caller (ALLOW/BLOCK/etc.), present
    for a quick "did this proxied call actually go through" filter without
    needing to join, not a duplicate source of truth for WHY.

    No prompt or response content is logged here either — same privacy
    discipline as every other audit event in this codebase. `usage` is
    whatever the provider returned (core.llm_providers.LLMResponse.usage),
    logged verbatim since it is already just token counts, not content.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "gateway_call",
        "request_id": request_id,
        "tenant": tenant,
        "capability": capability,
        "provider": provider,
        "model": model,
        "success": success,
        "latency_ms": latency_ms,
        "decision": decision,
        "usage": usage,
        "error": error,
    }

    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.info("Gateway Call", extra=log_entry)