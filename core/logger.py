# core/logger.py
import json
import hashlib
from datetime import datetime
import logging
from pythonjsonlogger import jsonlogger
import os

# --- CONFIGURE STRUCTURED LOGGING ---
logger = logging.getLogger("gatekeeper")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # JSON Audit Handler
    audit_handler = logging.FileHandler("audit.jsonl", encoding="utf-8")
    json_formatter = jsonlogger.JsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    audit_handler.setFormatter(json_formatter)
    
    audit_logger = logging.getLogger("gatekeeper.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False 
    audit_logger.addHandler(audit_handler)

def get_logger(name="gatekeeper"):
    return logging.getLogger(name)

def log_event(capability, prompt, risk, decision, metadata=None):
    if metadata is None: metadata = {}
    
    timestamp = datetime.now().isoformat()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    log_entry = {
        "timestamp": timestamp,
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