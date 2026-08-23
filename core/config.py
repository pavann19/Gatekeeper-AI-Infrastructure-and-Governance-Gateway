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

    # Run the fusion's transformer detectors concurrently rather than one
    # after another. Wall time becomes roughly the slowest detector instead
    # of the sum of all of them — but ONLY when the CPU has headroom to run
    # them at once. PyTorch already parallelises within a single forward
    # pass, so on a small core count this can oversubscribe the CPU and win
    # nothing (or lose). It is a flag, not a constant, so a deployment that
    # measures no gain — or measures a regression — can turn it off without
    # a redeploy. See scripts/benchmark_fusion_parallel.py for the
    # measurement on a given machine.
    FUSION_PARALLEL: bool = True

    # Score each prompt under a per-attack-class fusion policy rather than a
    # single global one, taking the most severe verdict. Measured out-of-fold
    # at matched union FPR (scripts/analyze_per_class_thresholds.py):
    # harmful_content recall 28.7% -> 31.5%, overall 82.8% -> 83.3%. A real
    # but modest gain — it does NOT solve harmful-content detection, which
    # needs a better instrument rather than better thresholds. Costs three
    # dot products over features already computed, so runtime impact is
    # negligible. Falls back to the global policy automatically when the
    # artifact has no per_class section (v1 artifacts).
    FUSION_PER_CLASS: bool = True

    # May a judge's SAFE verdict fully clear a prompt the deterministic layer
    # already flagged?
    #
    # FALSE (default) means no: the arbiter can confirm danger (HIGH) or
    # reduce to MEDIUM, but never returns a flagged prompt to LOW. This is the
    # "deterministic arbiter" posture — the LLM advises, the policy decides.
    #
    # This is a MEASURED default, not a philosophical one. On the 546-prompt
    # deepset benchmark the judge used its clear-authority 17 times: it cleared
    # 13 genuine attacks and 4 genuine benign prompts — a 3.25:1 losing trade.
    # Removing that authority moves recall 55.67% -> 62.07% and F1 0.671 ->
    # 0.712, for +1.17pp FPR (6.12% -> 7.29%).
    #
    # Historical note, because the code here is easy to misread: an earlier
    # `threat_present` tautology (§1n) made this cap unconditional by accident.
    # Fixing the tautology was correct — the expression really was always True
    # — but it silently handed the judge clear-authority and cost ~6 points of
    # recall. The cap is now an explicit, measured policy rather than an
    # accident, and `_in_upper_ambiguous_band` still governs which of the two
    # override sources is recorded for audit.
    JUDGE_MAY_CLEAR_TO_LOW: bool = False

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
    # "mistral" was the original default, stale since Llama Guard 3 became
    # the validated judge (docs/ENGINEERING_ASSESSMENT.md sections 1g/1j) --
    # docker-compose.yml already overrides this to "llama-guard3" for
    # container deployments, but the native/local default (no .env, no env
    # var override) was never updated to match, so a fresh local run silently
    # asked Ollama for a model this project stopped validating against.
    OLLAMA_MODEL: str = "llama-guard3"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"

    # --- Phase 5 (Real LLM Gateway), scoped down to provider abstraction
    # only for this pass — core/llm_providers.py. No request-forwarding
    # endpoint consumes these yet; they exist so a provider can be
    # constructed from configuration rather than only via explicit
    # constructor arguments (matching how OLLAMA_API_URL/OLLAMA_MODEL
    # already work). Empty-string API key defaults are deliberate: a
    # provider requiring one fails loudly at call time
    # (LLMProviderError), not at import time, so importing this module
    # never requires every provider to be configured.
    OLLAMA_CHAT_URL: str = "http://localhost:11434/api/chat"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = ""
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"
    ANTHROPIC_MODEL: str = ""
    ANTHROPIC_VERSION: str = "2023-06-01"

    # File Paths
    POLICY_FILE: str = "policies.json"
    POLICY_RULES_FILE: str = "policy_rules.json"

    # Where core/policy_versioning.py stores snapshots taken before each
    # policy deploy (Phase 3, Policy-as-Code: versioning and rollback).
    # Relative to cwd by default, matching AUDIT_LOG_PATH's own convention —
    # a container deployment should point this at the same mounted volume,
    # since a snapshot taken right before a container recreation is exactly
    # the one an operator would want to roll back to.
    POLICY_VERSIONS_DIR: str = "policy_versions"

    # core/review_queue.py's storage (Phase 4: Human Review). A single
    # mutable JSON file, not the audit log's append-only convention — see
    # that module's docstring for why. Same mounted-volume guidance as
    # AUDIT_LOG_PATH and POLICY_VERSIONS_DIR applies for a container
    # deployment: a pending review lost on container recreation is a review
    # that silently never gets resolved.
    REVIEW_QUEUE_FILE: str = "review_queue.json"

    # Path to the JSONL audit log. Relative to cwd by default — unchanged
    # from before this setting existed. A container deployment should point
    # this at a mounted volume, since the audit record is this project's
    # compliance artefact and must not be silently discarded on every
    # container recreation.
    AUDIT_LOG_PATH: str = "audit.jsonl"

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

    # Tenant Resolver (core/tenancy.py) — identity and SLA only, not policy.
    # See that module's docstring for the identity/policy split and why it
    # matters. A missing file means every caller resolves to a safe default
    # tenant; configuring tenants.json is opt-in.
    TENANTS_FILE: str = "tenants.json"

    # Comma-separated CORS origin allowlist. Credentialed cross-origin requests
    # are only enabled when this is NOT a wildcard.
    CORS_ORIGINS: str = "*"

    # --- Rate limiting (core/rate_limit.py) ---
    #
    # Authenticated callers are bucketed by their verified key_id; anonymous
    # callers by transport peer address. Anonymous gets a much smaller
    # allowance because it is unattributable, not because it is untrusted:
    # abuse from an anonymous caller cannot be traced to a revocable key.
    #
    # The defaults assume the measured cold-path cost of an assessment
    # (multi-second p95 — §1p). 120/min authenticated is roughly one request
    # every 500ms sustained, which a legitimate integration will not exceed
    # while a retry storm will.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_AUTHENTICATED_RPM: float = 120.0
    RATE_LIMIT_ANONYMOUS_RPM: float = 20.0

    # Burst window. Capacity = rate-per-second x this, so a 120/min key may
    # arrive with 20 requests at once and still be within budget, provided its
    # SUSTAINED rate stays under the limit. Bursts are normal client
    # behaviour; sustained overload is the thing worth rejecting.
    RATE_LIMIT_BURST_SECONDS: float = 10.0

    # Upper bound on distinct callers tracked at once. Prevents identity
    # rotation from exhausting memory; see the declared tradeoff in
    # core/rate_limit.py's module docstring.
    RATE_LIMIT_MAX_TRACKED: int = 10_000

    # Whether to believe X-Forwarded-For when identifying anonymous callers.
    # OFF by default and it must stay off unless a trusted proxy sits in
    # front: on a directly-exposed service, trusting this header lets any
    # caller mint unlimited identities and bypass the limit entirely.
    RATE_LIMIT_TRUST_FORWARDED_FOR: bool = False

    # --- Assessment execution bounds (api/main.py) ---
    #
    # asyncio.to_thread uses the default executor, which sizes itself to
    # min(32, cpu_count + 4). That is badly wrong here: each concurrent
    # assessment runs three transformer detectors, and PyTorch already
    # parallelises WITHIN each forward pass (§1m measured oversubscription at
    # only 3 concurrent models). Sixteen concurrent assessments would thrash
    # the CPU into uselessness rather than serving sixteen users. A small
    # bounded pool converts overload into queueing, which the timeout below
    # then bounds.
    ASSESS_MAX_CONCURRENCY: int = 4

    # Ceiling on how long a caller waits for an assessment, including time
    # spent queued for a worker. On expiry the request fails with 503 rather
    # than a fabricated verdict — see the rationale at the call site.
    ASSESS_TIMEOUT_SECONDS: float = 30.0

    # Load every model during startup instead of on the first request.
    #
    # This is NOT an optimisation, it is a correctness fix for the timeout
    # above. Cold model loading measured ~35s on the reference machine
    # (§1p's benchmark spent 34.6s on its first item), which is longer than
    # ASSESS_TIMEOUT_SECONDS — so with lazy loading the FIRST request after
    # every deploy is guaranteed to 503, and so is every request that arrives
    # while it is still loading. Warming up front moves that cost to boot,
    # where an orchestrator's readiness probe is designed to wait for it.
    WARM_MODELS_ON_STARTUP: bool = True

    # --- Observability (core/metrics.py) ---
    #
    # The Prometheus exposition endpoint. Metrics leak operational detail —
    # traffic volume, block rates, which tenants are active — so in a real
    # deployment this should be reachable only from the monitoring network.
    # METRICS_REQUIRE_AUTH exists for deployments that cannot segment the
    # network and would rather require a key; it defaults off because
    # requiring one breaks a default Prometheus scrape config, and a control
    # that silently stops your monitoring is worse than the exposure.
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"
    METRICS_REQUIRE_AUTH: bool = False

    # Correlation IDs. An inbound X-Request-ID is honoured so a trace can span
    # services, but it is caller-supplied data: it is length- and
    # charset-checked before use, because it lands in the audit log and an
    # unvalidated value is a log-injection primitive.
    REQUEST_ID_HEADER: str = "X-Request-ID"
    REQUEST_ID_MAX_LENGTH: int = 64

    @field_validator("RATE_LIMIT_AUTHENTICATED_RPM", "RATE_LIMIT_ANONYMOUS_RPM")
    @classmethod
    def _validate_rpm(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                "Rate limits must be positive. To disable limiting, set "
                "RATE_LIMIT_ENABLED=false rather than using a zero rate — an "
                "explicit off switch is far harder to misread than a 0."
            )
        return v

    @field_validator("ASSESS_MAX_CONCURRENCY")
    @classmethod
    def _validate_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("ASSESS_MAX_CONCURRENCY must be at least 1.")
        return v

    @field_validator("ASSESS_TIMEOUT_SECONDS")
    @classmethod
    def _validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ASSESS_TIMEOUT_SECONDS must be positive.")
        return v

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
POLICY_VERSIONS_DIR = settings.POLICY_VERSIONS_DIR