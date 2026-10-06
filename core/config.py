# core/config.py
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Capability Tiers
    CAPABILITY_GENERAL: str = "GENERAL"       # Formerly 'student'
    CAPABILITY_ELEVATED: str = "ELEVATED"     # Formerly 'researcher'
    CAPABILITY_INTERNAL: str = "INTERNAL"     # Formerly 'admin'

    SEMANTIC_THRESHOLD_HIGH: float = 0.48
    SEMANTIC_THRESHOLD_MEDIUM: float = 0.30
    META_INTENT_THRESHOLD: float = 0.30

    # Not yet calibrated — no labelled data for these decisions.
    EDUCATIONAL_THRESHOLD: float = 0.45
    DOMAIN_THRESHOLD: float = 0.22

    CACHE_SIMILARITY_THRESHOLD: float = 0.99

    FUSION_PARALLEL: bool = True

    FUSION_PER_CLASS: bool = True

    JUDGE_MAY_CLEAR_TO_LOW: bool = False

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

    HALLUCINATION_CHECK_ENABLED: bool = False

    # The optional local toxicity fallback improves availability when the
    # primary judge is down. Output release still requires an explicit SAFE
    # verdict; unavailable, malformed, timed-out and failed judgments BLOCK.
    OUTPUT_JUDGE_FALLBACK_ENABLED: bool = True
    # toxic_bert emits an independent sigmoid probability per label (see
    # TransformerDetector's multi_label=True path); 0.5 is the same neutral
    # midpoint threshold used to interpret any single sigmoid output before
    # a fitted policy exists for it, deliberately conservative rather than
    # tuned, since this path only ever runs during a primary-judge outage.
    OUTPUT_JUDGE_FALLBACK_THRESHOLD: float = 0.5

    # Execution Environment
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "llama-guard3"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    # Default encoder pin; custom encoders must supply their own revision.
    EMBEDDING_MODEL_REVISION: str | None = None

    # --- (Real LLM Gateway), scoped down to provider abstraction
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
    POLICY_FILE: str = "policies/threat_anchors.json"
    POLICY_RULES_FILE: str = "policy_rules.json"

    # Where core/policy_versioning.py stores snapshots taken before each
    # policy deploy (, Policy-as-Code: versioning and rollback).
    # Relative to cwd by default, matching AUDIT_LOG_PATH's own convention —
    # a container deployment should point this at the same mounted volume,
    # since a snapshot taken right before a container recreation is exactly
    # the one an operator would want to roll back to.
    POLICY_VERSIONS_DIR: str = "policy_versions"

    # Where local/private benchmark run results live for the Developer UI's
    # Benchmarks view (). Read-only from the API's side -- nothing in
    # this codebase writes here at request time. The public repo keeps summaries
    # in docs/ rather than committing raw _evidence artifacts.
    EVIDENCE_DIR: str = "_evidence"

    # core/review_queue.py's storage (Human Review). A single
    # mutable JSON file, not the audit log's append-only convention — see
    # that module's docstring for why. Same mounted-volume guidance as
    # AUDIT_LOG_PATH and POLICY_VERSIONS_DIR applies for a container
    # deployment: a pending review lost on container recreation is a review
    # that silently never gets resolved.
    REVIEW_QUEUE_FILE: str = "review_queue.json"
    REVIEW_MAX_RECORDS: int = 10000
    REVIEW_MAX_BYTES: int = 16777216

    # Path to the JSONL audit log. Relative to cwd by default — unchanged
    # from before this setting existed. A container deployment should point
    # this at a mounted volume, since the audit record is this project's
    # compliance artefact and must not be silently discarded on every
    # container recreation.
    AUDIT_LOG_PATH: str = "audit.jsonl"

    # Legacy configuration field retained for compatibility. Authoritative
    # audit failures always fail closed; False no longer enables un-audited traffic.
    AUDIT_WRITE_FAILS_CLOSED: bool = True

    # Embedded-SQLite mirror of the audit trail. Path to the SQLite
    # file (stdlib sqlite3, no new service). Same mounted-volume guidance as
    # AUDIT_LOG_PATH. When AUDIT_SQLITE_ENABLED is True (default), every
    # log_*_event dual-writes into this DB in addition to the JSONL.
    #
    # FAIL-MODE: the JSONL write
    # stays the AUTHORITATIVE persistence for now. A SQLite mirror-write
    # failure is logged CRITICAL and swallowed — it does NOT fail the request
    # and does NOT trip AUDIT_WRITE_FAILS_CLOSED, because the JSONL record
    # (which that flag guards) still succeeded. Reads still come from the
    # JSONL via core/activity.py; a later increment flips reads to SQLite and
    # only then does a SQLite failure become request-fatal.
    AUDIT_DB_PATH: str = "audit.db"
    AUDIT_SQLITE_ENABLED: bool = True

    # When True, the activity feed / trace / raw-logs
    # reads (core/activity.py) are served from the SQLite store instead of the
    # JSONL reverse-tail scan, and a SQLite mirror-write failure becomes
    # request-fatal (fails CLOSED via AUDIT_WRITE_FAILS_CLOSED) because the DB
    # is now the authoritative read path. Default False: the capability ships
    # tested and one flag away, but making SQLite the read authority is a
    # deployment decision, not an automatic one. A read error still falls back
    # to the JSONL scan so enabling this cannot black out the feed.
    AUDIT_READ_FROM_SQLITE: bool = False

    # /health probes the judge backend at most once per this many seconds;
    # in between it serves the last probe result. Stops a burst of health
    # checks from each paying the full connect/read timeout when the backend
    # is unreachable (the residual after the circuit-breaker short-circuit,
    # which only helps once the breaker has actually opened).
    HEALTH_JUDGE_PROBE_TTL_SECONDS: float = 10.0

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

    ASSESS_MAX_CONCURRENCY: int = 4

    TORCH_INTRA_OP_THREADS: int = 1
    TORCH_INTER_OP_THREADS: int = 1

    # Ceiling on how long a caller waits for an assessment, including time
    # spent queued for a worker. On expiry the request fails with 503 rather
    # than a fabricated verdict — see the rationale at the call site.
    ASSESS_TIMEOUT_SECONDS: float = 30.0

    # --- Gateway execution bounds (api/main.py, request forwarding) ---
    #
    # A SEPARATE pool from ASSESS_MAX_CONCURRENCY, deliberately — an LLM
    # completion is network I/O to an external provider, not local
    # transformer inference, so it competes for a fundamentally different
    # resource (outbound connections/provider rate limits, not this
    # machine's CPU). Sharing the assess pool would let a slow external
    # provider starve local risk assessment of workers, which is exactly
    # the coupling a bounded-and-separate pool avoids.
    GATEWAY_MAX_CONCURRENCY: int = 4

    # Higher than ASSESS_TIMEOUT_SECONDS on purpose: a non-streaming LLM
    # completion routinely takes longer than this project's own local
    # detector pipeline, especially from a larger or remote model. Still a
    # hard ceiling, not a suggestion — the same fail-closed-to-503
    # reasoning as ASSESS_TIMEOUT_SECONDS applies: a timeout here means the
    # call did not happen and must not be treated as if it did.
    GATEWAY_TIMEOUT_SECONDS: float = 60.0

    # Which LLMProvider (core/llm_providers.py) a gateway call uses when
    # the caller doesn't specify one explicitly.
    LLM_GATEWAY_DEFAULT_PROVIDER: str = "ollama"

    # Ordered, comma-separated fallback providers tried in sequence when the
    # one actually selected for this call fails or times out (    # "Timeout/failure handling, fallback"). Same comma-separated-string
    # convention as CORS_ORIGINS, parsed the same way at the call site.
    #
    # Applies ONLY when the caller did not explicitly name a provider. A
    # caller who names "openai_compatible" chose that provider on purpose —
    # silently substituting a different one on failure would violate that
    # choice, the same reasoning AssessRequest.model_config's extra="forbid"
    # applies elsewhere in this codebase: an explicit input is never
    # second-guessed by the gateway. Fallback exists for the DEFAULT-provider
    # path, where "some working provider" is what the caller actually wants.
    GATEWAY_FALLBACK_PROVIDERS: str = ""

    # --- Token accounting ("Token accounting, model selection") ---
    #
    # Enforced PAST the fact, not predicted ahead of it: a provider's own
    # response is the only place token usage is known, so this can only ever
    # reject the NEXT call once a tenant's tracked usage has already crossed
    # the line, not the call that put them over. That is the same shape as
    # billing-usage caps on every commercial LLM API and is stated here so it
    # is not mistaken for a pre-flight cost estimate this gateway does not
    # (and cannot, without calling the provider first) perform.
    GATEWAY_TOKEN_QUOTA_ENABLED: bool = False

    # Applies to a tenant with no explicit TenantConfig.token_quota_daily
    # override (core/tenancy.py). 0 means "unlimited" for that tenant even
    # while enforcement is enabled globally — same "0 disables, don't fake it
    # with a huge number" convention as the rate limiter's own settings.
    GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT: int = 0

    WARM_MODELS_ON_STARTUP: bool = True

    # Registers core/demo_tools.py's four sandboxed demo tools at
    # startup, so POST /api/v1/tools/call has something real to call
    # without a deployment writing its own tools first. Defaults OFF: a
    # production deployment should not get demo tools just because this
    # module happens to be importable — the same "opt-in, not automatic"
    # contract core/demo_tools.py::register_demo_tools already documents.
    REGISTER_DEMO_TOOLS: bool = False

    REGISTER_REAL_TOOLS: bool = False

    # Exact-match hostname allowlist for core/real_tools.py's "http.get"
    # tool -- the first REAL (non-sandboxed) tool this project ships,
    # making an actual outbound network call. Empty by default: fail
    # closed, the same "0/empty disables, don't fake a permissive
    # default" convention GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT already uses.
    # A deployment wanting this tool usable must explicitly list every
    # hostname it may reach — comma-separated, e.g.
    # "docs.python.org,api.github.com". Subdomains are NOT implicitly
    # covered by a parent domain (no "*.example.com" wildcard support):
    # an explicit list is easier to audit than a wildcard-matching rule
    # that could be misread as narrower than it actually is.
    TOOL_HTTP_GET_ALLOWED_DOMAINS: str = ""

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

    @field_validator("GATEWAY_MAX_CONCURRENCY")
    @classmethod
    def _validate_gateway_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("GATEWAY_MAX_CONCURRENCY must be at least 1.")
        return v

    @field_validator("GATEWAY_TIMEOUT_SECONDS")
    @classmethod
    def _validate_gateway_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("GATEWAY_TIMEOUT_SECONDS must be positive.")
        return v

    @field_validator("GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT")
    @classmethod
    def _validate_token_quota_default(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                "GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT must be >= 0. Use 0 for "
                "unlimited rather than a negative sentinel."
            )
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

    # Detector Model Integrity Verification
    DETECTOR_VERIFY_HASHES: bool = False

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
DETECTOR_VERIFY_HASHES = settings.DETECTOR_VERIFY_HASHES
