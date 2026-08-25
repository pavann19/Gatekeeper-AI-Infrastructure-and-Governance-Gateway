import re
import spacy

from core.config import SPACY_MODEL
from core.logger import get_logger

logger = get_logger(__name__)

# --- CONFIGURATION ---
# Load spaCy small model for efficiency (Research Grade: Lightweight)
try:
    NLP_MODEL = spacy.load(SPACY_MODEL)
    # Disable heavy pipeline components we don't need (parser, lemmatizer)
    # to keep latency under 20ms.
    NLP_MODEL.disable_pipes(["parser", "tagger", "lemmatizer", "attribute_ruler"])
except OSError:
    logger.warning(f"spaCy model '{SPACY_MODEL}' not found. Run: python -m spacy download {SPACY_MODEL}")
    NLP_MODEL = None

# 1. DETERMINISTIC PATTERNS (The "Fast Path")
# Standard International & Indian formats
REGEX_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "PHONE": r'(?:\+91[\-\s]?)?[6-9]\d{9}\b|(?:\+1[\-\s]?)?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}',
    "IP_ADDR": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "AADHAAR": r'\b\d{4}\s\d{4}\s\d{4}\b'
}

# 2. CONTEXTUAL ENTITIES (The "Slow Path")
# Only redact high-confidence entities relevant to safety
NER_LABELS = {"PERSON", "ORG", "GPE"} 

def redact_pii(text: str, tenant_config=None) -> tuple:
    """
    Hybrid PII Detection Pipeline with optional per-tenant overrides.
    Strategy: Fail-Fast. 
    1. Run Regex (skipping any patterns disabled for this tenant).
    2. If clean, run NER for configured labels (default: PERSON, ORG, GPE).
    """
    clean_text = text
    detected_items = []
    detection_source = "NONE"

    disabled_regex = set()
    ner_labels = NER_LABELS
    if tenant_config is not None:
        if getattr(tenant_config, "privacy_disabled_patterns", None):
            disabled_regex = set(tenant_config.privacy_disabled_patterns)
        if getattr(tenant_config, "privacy_ner_labels", None) is not None:
            ner_labels = set(tenant_config.privacy_ner_labels)

    # --- STAGE 1: SYMBOLIC (Regex) ---
    regex_hit = False
    for label, pattern in REGEX_PATTERNS.items():
        if label in disabled_regex:
            continue
        matches = re.findall(pattern, clean_text)
        if matches:
            regex_hit = True
            detection_source = "REGEX_FAST"
            for match in matches:
                mask = f"[REDACTED:{label}]"
                clean_text = clean_text.replace(match, mask)
                detected_items.append(f"{label}:{match}")

    # OPTIMIZATION: If Regex found something, we assume the prompt 
    # is "dirty" and skip the expensive NER model to save ~200ms.
    if regex_hit:
        return clean_text, {"pii_found": True, "source": detection_source, "items": detected_items}

    # --- STAGE 2: NEURAL (spaCy NER) ---
    if NLP_MODEL and ner_labels:
        doc = NLP_MODEL(clean_text)
        ner_hit = False
        
        # We iterate in reverse to avoid index shifting issues during replacement
        for ent in reversed(doc.ents):
            if ent.label_ in ner_labels:
                ner_hit = True
                detection_source = "NER_CONTEXT"
                mask = f"[REDACTED:{ent.label_}]"
                
                # Replace string slice safely
                clean_text = clean_text[:ent.start_char] + mask + clean_text[ent.end_char:]
                detected_items.append(f"{ent.label_}:{ent.text}")

        if ner_hit:
            return clean_text, {"pii_found": True, "source": detection_source, "items": detected_items}

    return clean_text, {"pii_found": False, "source": "CLEAN", "items": []}