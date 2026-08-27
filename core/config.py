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

    # Output hallucination check (core.output_guardrails.check_semantic_grounding).
    # Defaults OFF: a live pentest (docs/ROADMAP_V2.md's Phase 8 findings)
    # found it compares OUTPUT text against `educational_store`'s centroid --
    # which is actually the INPUT-side educational-context anchor set
    # (core/risk.py's EDUCATIONAL_CONTEXT_ANCHORS, populated as a side
    # effect of prior /assess calls), a category error. Once that store is
    # warm, any legitimate response that isn't itself educational-content-
    # shaped -- a support reply, code, a factual answer -- risks a false
    # "hallucination" BLOCK. Same shape of mistake as DOMAIN_GUARDRAIL_MODE
    # above: a real check, wired to the wrong reference corpus, left
    # disabled until a real "acceptable assistant output" corpus exists to
    # replace the borrowed one.
    HALLUCINATION_CHECK_ENABLED: bool = False

    # Output-judge fallback ("inverter") — core.semantic_judge.output_judge.
    # WHY THIS EXISTS: when the primary judge (Ollama/Llama Guard) is
    # unreachable, output_judge() returns the sentinel "JUDGE_OFFLINE", and
    # core.output_guardrails.assess_output only ever checks
    # `if verdict == "DANGEROUS"` -- anything else, including
    # "JUDGE_OFFLINE", falls through to ALLOW. That is a silent fail-OPEN:
    # exactly the window an attacker would want, and exactly the outage a
    # real deployment WILL eventually hit (Ollama restarts, OOMs, or the
    # host it runs on reboots). Fix: when the primary judge is unreachable,
    # fall back to `toxic_bert` -- already loaded and warmed at startup for
    # the INPUT-side fusion ensemble (core/fusion.py's LIVE_MODEL_DETECTORS,
    # chosen there specifically for being fast enough for synchronous
    # request handling, unlike Llama Guard) -- instead of allowing blind.
    # It is not as accurate as the primary judge, but a fast, local,
    # network-independent second opinion beats no opinion at all: the same
    # reasoning as an inverter picking up a house's critical circuits the
    # instant utility power drops, not a substitute for the grid but enough
    # to keep the lights from going out entirely. True JUDGE_OFFLINE (fail
    # open) is now reached only if THIS fallback model also fails to load.
    OUTPUT_JUDGE_FALLBACK_ENABLED: bool = True
    # toxic_bert emits an independent sigmoid probability per label (see
    # TransformerDetector's multi_label=True path); 0.5 is the same neutral
    # midpoint threshold used to interpret any single sigmoid output before
    # a fitted policy exists for it, deliberately conservative rather than
    # tuned, since this path only ever runs during a primary-judge outage.
    OUTPUT_JUDGE_FALLBACK_THRESHOLD: float = 0.5

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

    # Where real benchmark run results (accuracy/latency/confusion-matrix
    # JSON reports produced by this project's own benchmark scripts) live,
    # for the Developer UI's Benchmarks view (Phase 7). Read-only from the
    # API's side -- nothing in this codebase writes here at request time.
    EVIDENCE_DIR: str = "_evidence"

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

    # --- Gateway execution bounds (api/main.py, Phase 5: request forwarding) ---
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
    # one actually selected for this call fails or times out (Phase 5:
    # "Timeout/failure handling, fallback"). Same comma-separated-string
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

    # --- Token accounting (Phase 5: "Token accounting, model selection") ---
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

    # Registers core/demo_tools.py's four sandboxed demo tools at
    # startup, so POST /api/v1/tools/call has something real to call
    # without a deployment writing its own tools first. Defaults OFF: a
    # production deployment should not get demo tools just because this
    # module happens to be importable — the same "opt-in, not automatic"
    # contract core/demo_tools.py::register_demo_tools already documents.
    REGISTER_DEMO_TOOLS: bool = False

    # Registers core/real_tools.py's "http.get" -- the only REAL (non-
    # sandboxed, live-network-calling) tool this project ships -- into the
    # tool registry at startup. Defaults OFF, same opt-in contract as
    # REGISTER_DEMO_TOOLS above and as register_real_tools()'s own
    # docstring already states. HONEST NOTE: this flag was ADDED during a
    # live pentest (docs/ROADMAP_V2.md's Phase 8 findings) after
    # discovering register_real_tools() existed, was fully tested
    # (tests/test_real_tools.py, tests/test_real_tools_edge_cases.py --
    # including real SSRF-protection coverage), and was STILL NEVER WIRED
    # UP anywhere -- no setting gated it, no startup call invoked it, so
    # the tool was unreachable in every real deployment despite being
    # production-ready. This is the fix for that gap, not a new feature.
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