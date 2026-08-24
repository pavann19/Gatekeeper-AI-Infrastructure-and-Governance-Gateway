# core/policy_loader.py
"""
Centralized Policy Loader — single module for loading all JSON policy files.
Loaded once at import time. Safe fallback to empty lists on missing/invalid files.
"""
import json
import os

from core.logger import get_logger

logger = get_logger(__name__)

# --- Policy File Paths ---
DOMAIN_ANCHORS_FILE = "policies/domain_anchors.json"
SYMBOLIC_RULES_FILE = "policies/symbolic_rules.json"

# --- Internal State (loaded once) ---
_domain_anchors = []
_suspicious_phrases = []
_jailbreak_patterns = None            # None signals fail-closed
_instruction_override_patterns = None  # None signals fail-closed
_hard_ban_keywords = None             # None signals fail-closed


def _load_json_file(filepath):
    """Loads and returns parsed JSON data from a file.
    Returns None if file is missing or invalid."""
    if not os.path.exists(filepath):
        logger.warning(f"{filepath} not found.")
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def _init_policies():
    """Loads all policy files into module-level variables at import time."""
    global _domain_anchors, _suspicious_phrases, _jailbreak_patterns
    global _instruction_override_patterns, _hard_ban_keywords

    # --- Domain Anchors ---
    data = _load_json_file(DOMAIN_ANCHORS_FILE)
    if data is not None:
        _domain_anchors = data.get("domains", [])
    else:
        logger.warning("Domain guardrail disabled (no anchors loaded).")
        _domain_anchors = []

    # --- Symbolic Rules ---
    data = _load_json_file(SYMBOLIC_RULES_FILE)
    if data is not None:
        _suspicious_phrases = data.get("suspicious_phrases", [])
        _jailbreak_patterns = data.get("jailbreak_patterns", [])
        _instruction_override_patterns = data.get("instruction_override_patterns", [])
        _hard_ban_keywords = data.get("hard_ban_keywords", [])
    else:
        logger.error("Symbolic rules not loaded. Symbolic detection will fail closed.")
        _suspicious_phrases = []
        _jailbreak_patterns = None
        _instruction_override_patterns = None
        _hard_ban_keywords = None


# --- Public Accessor Functions ---

def get_domain_anchors():
    """Returns list of domain anchor strings."""
    return _domain_anchors

def get_suspicious_phrases():
    """Returns list of suspicious phrase strings."""
    return _suspicious_phrases

def get_jailbreak_patterns():
    """Returns list of jailbreak (persona/roleplay hijack) regex pattern
    strings, or None if load failed. Distinct from
    get_instruction_override_patterns() -- see policies/symbolic_rules.json's
    _comment for why these were split."""
    return _jailbreak_patterns

def get_instruction_override_patterns():
    """Returns list of instruction-override (prompt injection) regex pattern
    strings, or None if load failed."""
    return _instruction_override_patterns

def get_hard_ban_keywords():
    """Returns list of hard ban keyword strings, or None if load failed."""
    return _hard_ban_keywords


# Load all policies once at import time
_init_policies()
