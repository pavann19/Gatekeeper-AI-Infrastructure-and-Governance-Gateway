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
        "dynamic_threat_score": metadata.get("dynamic_threat_score", None)
    }

    # Structured JSON log for Audit
    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.info("Governance Decision", extra=log_entry)