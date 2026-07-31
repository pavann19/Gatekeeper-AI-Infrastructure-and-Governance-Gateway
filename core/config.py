# core/config.py
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Capability Tiers
    CAPABILITY_GENERAL: str = "GENERAL"       # Formerly 'student'
    CAPABILITY_ELEVATED: str = "ELEVATED"     # Formerly 'researcher'
    CAPABILITY_INTERNAL: str = "INTERNAL"     # Formerly 'admin'

    # --- Security Thresholds ---
    #
    # SEMANTIC_THRESHOLD_MEDIUM and META_INTENT_THRESHOLD are CALIBRATED, not
    # chosen by hand. Selected by scripts/calibrate_thresholds.py as the point
    # maximising recall subject to a 5% false-positive-rate budget, fitted on a
    # 382-prompt calibration split and verified on a held-out 164-prompt split:
    #
    #   calibration : recall 31.72%  precision 80.70%  FPR 4.64%
    #   holdout     : recall 25.86%  precision 88.24%  FPR 1.89%
    #   ROC AUC     : 0.745 (deterministic signals, judge excluded)
    #
    # Evidence: _evidence/calibration_report.json, _evidence/roc_curve.png
    # Re-run calibration after ANY change to the anchors or the encoder.
    #
    # Recall is low because the encoder is English-only while the evaluation set
    # is largely German. That is an encoder limitation, not a threshold one —
    # see docs/ENGINEERING_ASSESSMENT.md section 1b.
    SEMANTIC_THRESHOLD_HIGH: float = 0.48
    SEMANTIC_THRESHOLD_MEDIUM: float = 0.30
    META_INTENT_THRESHOLD: float = 0.30

    # Not yet calibrated — no labelled data for these decisions.
    EDUCATIONAL_THRESHOLD: float = 0.45
    DOMAIN_THRESHOLD: float = 0.22

    # CALIBRATED via scripts/diagnose_cache_threshold.py against
    # deepset/prompt-injections: at the old default (0.95), 9.1% of
    # near-duplicate prompt pairs above threshold had OPPOSITE ground-truth
    # labels; at 0.98, 50% did. Sentence-embedding cosine similarity measures
    # bulk topical content, not the presence of a short adversarial clause, so
    # no threshold low enough to matter is provably safe for this cache's
    # fuzzy-match tier. 0.99 is the highest value with zero observed unsafe
    # pairs in that measurement — a materially safer margin, not a guarantee.
    # core/cache.py checks an exact prompt-hash match FIRST and unconditionally
    # (zero collision risk); this threshold only governs the fuzzy fallback.
    CACHE_SIMILARITY_THRESHOLD: float = 0.99

    # Domain (topicality) guardrail posture.
    #   "off"       — do not evaluate topicality at all (default).
    #   "advisory"  — report topicality in the response; does not affect risk.
    #   "enforcing" — off-domain prompts are escalated to MEDIUM risk.
    #
    # Defaults to "off" because the shipped domain corpus describes THIS
    # project's subject area.  A third-party deployment must supply its own
    # corpus before enabling enforcement, otherwise every benign prompt
    # outside ML/security topics is flagged.  Leaving this on by default was
    # the cause of the 98% false-positive rate in the pre-calibration
    # benchmark — see docs/ENGINEERING_ASSESSMENT.md §1.
    DOMAIN_GUARDRAIL_MODE: str = "off"

    @field_validator("DOMAIN_GUARDRAIL_MODE")
    @classmethod
    def _validate_domain_mode(cls, v: str) -> str:
        allowed = {"off", "advisory", "enforcing"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(
                f"DOMAIN_GUARDRAIL_MODE must be one of {sorted(allowed)}, got {v!r}"
            )
        return normalized
    
    # Execution Environment
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "mistral"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    
    # File Paths
    POLICY_FILE: str = "policies.json"
    POLICY_RULES_FILE: str = "policy_rules.json"

    # --- Authentication ---
    # Capability is resolved from a verified API key, never from the request
    # body. See core/auth.py for the vulnerability this replaces.
    #
    #   "optional" — anonymous requests are served at GENERAL (least privilege).
    #   "required" — anonymous requests are rejected with 401.
    #
    # Defaults to "optional" because GENERAL is already the safe tier: an
    # unauthenticated caller can never escalate, only be served conservatively.
    AUTH_MODE: str = "optional"
    API_KEYS_FILE: str = "api_keys.json"

    # Comma-separated CORS origin allowlist. Credentialed cross-origin requests
    # are only enabled when this is NOT a wildcard.
    CORS_ORIGINS: str = "*"

    @field_validator("AUTH_MODE")
    @classmethod
    def _validate_auth_mode(cls, v: str) -> str:
        allowed = {"optional", "required"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"AUTH_MODE must be one of {sorted(allowed)}, got {v!r}")
        return normalized

    # Dependency Models
    SPACY_MODEL: str = "en_core_web_sm"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# For backwards compatibility with existing imports:
CAPABILITY_GENERAL = settings.CAPABILITY_GENERAL
CAPABILITY_ELEVATED = settings.CAPABILITY_ELEVATED
CAPABILITY_INTERNAL = settings.CAPABILITY_INTERNAL
SEMANTIC_THRESHOLD_HIGH = settings.SEMANTIC_THRESHOLD_HIGH
SEMANTIC_THRESHOLD_MEDIUM = settings.SEMANTIC_THRESHOLD_MEDIUM
EDUCATIONAL_THRESHOLD = settings.EDUCATIONAL_THRESHOLD
DOMAIN_THRESHOLD = settings.DOMAIN_THRESHOLD
CACHE_SIMILARITY_THRESHOLD = settings.CACHE_SIMILARITY_THRESHOLD
META_INTENT_THRESHOLD = settings.META_INTENT_THRESHOLD

# New configuration properties exported directly:
OLLAMA_API_URL = settings.OLLAMA_API_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
SPACY_MODEL = settings.SPACY_MODEL
POLICY_FILE = settings.POLICY_FILE
POLICY_RULES_FILE = settings.POLICY_RULES_FILE