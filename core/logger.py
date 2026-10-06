# core/logger.py
import hashlib
import json
import os
from datetime import datetime
import logging
import threading

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

def get_logger(name="gatekeeper"):
    return logging.getLogger(name)


def _mirror_to_sqlite(event_type, log_entry):
    """Dual-write: also persist this audit record into the embedded
    SQLite store. The JSONL write above stays AUTHORITATIVE — a mirror
    failure here is logged CRITICAL and swallowed so it never fails the
    request or trips AUDIT_WRITE_FAILS_CLOSED (which guards the JSONL write,
    already done by the time this runs). See core/audit_store.py.
    """
    if not settings.AUDIT_SQLITE_ENABLED:
        return
    try:
        from core import audit_store  # lazy: audit_store imports from this module
        audit_store.write(event_type, log_entry)
    except Exception as e:  # noqa: BLE001
        # When reads are served from SQLite it is the authoritative store, so a
        # mirror failure means the record is not durably persisted — let it
        # propagate to api.main._write_audit_or_fail, which fails the request
        # CLOSED. Otherwise the JSONL write (already done) is authoritative and
        # this is a swallow-and-log.
        if getattr(settings, "AUDIT_READ_FROM_SQLITE", False):
            logging.getLogger("gatekeeper").critical(
                "SQLite audit write failed (%s: %s) and SQLite is the read "
                "authority — failing request CLOSED. event_type=%s request_id=%s",
                type(e).__name__, e, event_type, log_entry.get("request_id"),
            )
            raise
        logging.getLogger("gatekeeper").critical(
            "SQLite audit mirror failed (%s: %s) — JSONL record is authoritative, "
            "continuing. event_type=%s request_id=%s",
            type(e).__name__, e, event_type, log_entry.get("request_id"),
        )

class AuditPersistenceError(RuntimeError):
    """The authoritative audit store did not acknowledge persistence."""


_audit_lock = threading.Lock()


def _persist_audit(event_type, entry):
    from core.policy import policy_identity
    entry = {**entry, 'policy_digest': policy_identity()}
    sqlite_authority = settings.AUDIT_READ_FROM_SQLITE
    try:
        if sqlite_authority:
            if not settings.AUDIT_SQLITE_ENABLED:
                raise AuditPersistenceError("SQLite audit authority is disabled")
            _mirror_to_sqlite(event_type, entry)
        payload = json.dumps(entry, ensure_ascii=True) + "\n"
        try:
            with _audit_lock:
                path = settings.AUDIT_LOG_PATH
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "a", encoding="utf-8") as stream:
                    if stream.write(payload) != len(payload):
                        raise OSError("Incomplete audit write")
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception:
            if not sqlite_authority:
                raise
            logger.critical("JSONL audit mirror failed after authoritative SQLite persistence")
        if not sqlite_authority:
            _mirror_to_sqlite(event_type, entry)
    except Exception as exc:
        raise AuditPersistenceError("Authoritative audit persistence failed") from exc
    # Diagnostic observers run only after authoritative persistence succeeded.
    try:
        logging.getLogger("gatekeeper.audit").info("Audit persisted", extra=entry)
    except Exception:
        logger.warning("Diagnostic audit observer failed")


def log_event(capability, prompt, risk, decision, metadata=None):
    if metadata is None:
        metadata = {}

    timestamp = datetime.now().isoformat()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    log_entry = {
        "timestamp": timestamp,
        # Distinguishes this record from log_output_event's records in the
        # same audit stream — see that function's docstring (, Output
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
    _persist_audit("input_assessment", log_entry)


def log_output_event(capability, response_text, decision, metadata=None, tenant="unset", request_id="unset"):
    """
        Audit record for an OUTPUT assessment — distinct from log_event's INPUT
        record (implementation).

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
        "toxicity_assessed": metadata.get("toxicity_assessed", False),
        "toxicity_status": metadata.get("toxicity_status", "NOT_ASSESSED"),
        "hallucination_detected": metadata.get("hallucination_detected", False),
        "system_prompt_leak_detected": metadata.get("system_prompt_leak_detected", False),
        "source": metadata.get("source", "unknown"),
    }

    _persist_audit("output_assessment", log_entry)


def log_gateway_event(capability, provider, model, success, latency_ms, decision,
                      usage=None, error=None, tenant="unset", request_id="unset"):
    """Persist a distinct provider-gateway event with sanitized metadata."""
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

    _persist_audit("gateway_call", log_entry)


def log_tool_event(capability, tool_name, decision, risk_level, reason,
                   arguments=None, success=None, error=None,
                   tenant="unset", request_id="unset", phase="decision"):
    """Persist a tool intent/completion/failure event with argument hashes."""
    arguments_hash = None
    if arguments is not None:
        canonical = json.dumps(arguments, sort_keys=True, default=str)
        arguments_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "tool_call",
        "phase": phase,
        "request_id": request_id,
        "tenant": tenant,
        "capability": capability,
        "tool": tool_name,
        "decision": decision,
        "risk_level": risk_level,
        "reason": reason,
        "arguments_hash": arguments_hash,
        "success": success,
        "error": error,
    }

    _persist_audit("tool_call", log_entry)
