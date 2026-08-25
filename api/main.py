from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
import hashlib
import json
import os
import tempfile
import time
import asyncio
import functools
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from core import metrics
from api.schemas import (
    AssessRequest, AssessResponse, AssessOutputRequest, AssessOutputResponse,
    ReviewStatusResponse, ReviewResolveRequest, GatewayChatRequest, GatewayChatResponse,
    ToolCallRequest, ToolCallResponse, WhoAmIResponse,
    PolicyContentRequest, PolicyRollbackRequest,
)
from core.activity import find_by_request_id, get_recent_activity
from core.benchmarks import list_benchmark_runs
from core.policy_versioning import deploy_policy, list_versions, rollback_to
from core.privacy import NER_LABELS, REGEX_PATTERNS, redact_pii
from core.auth import auth_required, resolve_principal
from core.rate_limit import assess_rate_limiter, bucket_parameters
from core.tenancy import resolve_tenant
from core.risk import assess_risk
from core.cache import flush_cache
from core.logger import get_logger, log_event, log_output_event, log_gateway_event
from core.policy import get_default_action, load_policy_file, policy_decision, resolve_policy_set, validate_policy_file
from core.review_queue import enqueue_review, get_review, list_pending_reviews, resolve_review
from core.tools import execute_tool, get_tool_registry
from core.llm_providers import get_provider, list_provider_names, LLMProviderError
from core.token_quota import gateway_token_quota, extract_total_tokens, seconds_until_utc_midnight
from core.config import settings, CAPABILITY_INTERNAL

logger = get_logger("gatekeeper.api")

app = FastAPI(
    title="Gatekeeper AI Governance API",
    description="Neuro-Symbolic AI Security Gateway",
    version="2.0.0"
)

# Phase 7: the client UI (login, review dashboard, and whatever else is
# added later) is a set of static pages -- no build step, no new frontend
# dependency -- that talk to this same API over fetch() with an
# operator-supplied API key. Mounted here rather than run as a separate
# process so it always points at the API it's actually driving, with no
# separate deployment/CORS story to keep in sync. `ui/web_app.py`, the
# pre-existing Streamlit prototype, is not part of this mount (it is a
# separate `streamlit run` process, not a static asset) and is served here
# only incidentally as an inert, non-executed file if requested by name.
_UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(_UI_DIR):
    app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="client_ui")

# CORS. The previous configuration paired allow_origins=["*"] with
# allow_credentials=True, which browsers reject outright per the CORS spec and
# which would be unsafe if honoured — any origin could drive authenticated
# requests. Credentials are only permitted against an explicit origin allowlist.
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
_allow_credentials = "*" not in _cors_origins
if not _allow_credentials:
    logger.warning(
        "CORS is configured with a wildcard origin, so credentialed "
        "cross-origin requests are disabled. Set CORS_ORIGINS to an explicit "
        "comma-separated allowlist to permit them."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Assessment execution bounds
# ---------------------------------------------------------------------------
#
# A DEDICATED, SMALL pool rather than asyncio.to_thread's default executor.
# The default sizes itself to min(32, cpu_count + 4), which is actively harmful
# here: every concurrent assessment loads three transformer detectors, and
# PyTorch already parallelises within each forward pass. §1m measured the CPU
# as oversubscribed at just THREE concurrent models — sixteen would thrash
# rather than serve. Bounding the pool converts overload into a queue, and the
# timeout below bounds the queue.
_assess_pool = ThreadPoolExecutor(
    max_workers=settings.ASSESS_MAX_CONCURRENCY,
    thread_name_prefix="assess",
)


async def _run_bounded(func, *args):
    """
    Runs a blocking assessment on the bounded pool, with a total deadline that
    covers BOTH queueing and execution.

    HONEST LIMITATION, stated because it is easy to assume otherwise: Python
    cannot cancel a running thread. On timeout the caller stops waiting, but
    the worker keeps going until it finishes on its own. This bounds the
    CLIENT's wait, not the work. What actually bounds the work is the pool
    size above — a stuck assessment costs one of N workers and cannot multiply.
    """
    loop = asyncio.get_running_loop()
    # Incremented around the whole call, queueing included: the useful
    # question is "how many callers are waiting on this pool", not "how many
    # threads are busy". The former shows saturation before the timeout does.
    metrics.assessments_in_flight.inc()
    try:
        future = loop.run_in_executor(_assess_pool, functools.partial(func, *args))
        return await asyncio.wait_for(future, timeout=settings.ASSESS_TIMEOUT_SECONDS)
    finally:
        metrics.assessments_in_flight.dec()


# Separate pool from _assess_pool (Phase 5: Real LLM Gateway) — see
# core/config.py's GATEWAY_MAX_CONCURRENCY docstring for why sharing the
# assessment pool with proxied provider calls would be the wrong coupling.
_gateway_pool = ThreadPoolExecutor(
    max_workers=settings.GATEWAY_MAX_CONCURRENCY,
    thread_name_prefix="gateway",
)


async def _run_gateway_bounded(func, *args):
    """Same shape as `_run_bounded`, on the separate gateway pool and with
    GATEWAY_TIMEOUT_SECONDS instead of ASSESS_TIMEOUT_SECONDS — a
    non-streaming LLM completion routinely needs more headroom than this
    project's own local detector pipeline."""
    loop = asyncio.get_running_loop()
    metrics.gateway_calls_in_flight.inc()
    try:
        future = loop.run_in_executor(_gateway_pool, functools.partial(func, *args))
        return await asyncio.wait_for(future, timeout=settings.GATEWAY_TIMEOUT_SECONDS)
    finally:
        metrics.gateway_calls_in_flight.dec()


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")


def _resolve_request_id(request: Request) -> str:
    """
    Honours an inbound correlation ID, or mints one.

    An upstream service's ID is accepted so a trace can span systems — but it
    is caller-supplied and it lands in the audit log, so it is validated
    first. An unvalidated value here would be a log-injection primitive:
    newlines and control characters could forge additional audit records in a
    JSONL file that downstream tooling parses line by line. Anything failing
    the charset or length check is silently replaced rather than rejected,
    since a malformed trace header is not a reason to fail a security
    assessment.
    """
    supplied = request.headers.get(settings.REQUEST_ID_HEADER, "").strip()
    if (
        supplied
        and len(supplied) <= settings.REQUEST_ID_MAX_LENGTH
        and _REQUEST_ID_PATTERN.match(supplied)
    ):
        return supplied
    return uuid.uuid4().hex


def _route_template(request: Request) -> str:
    """
    The matched route's path template, for use as a metric label.

    NEVER the raw path. Labelling by raw path lets any caller create unlimited
    time series by requesting random URLs, which is a memory-exhaustion vector
    against the monitoring system rather than a reporting inconvenience.
    Unmatched requests collapse into one bucket.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def _client_address(request: Request) -> str:
    """
    Transport peer address, used to bucket anonymous callers.

    X-Forwarded-For is consulted ONLY when explicitly trusted via config,
    because on a directly-exposed service that header is caller-controlled and
    would hand out unlimited rate-limit identities. When trusted, the RIGHTMOST
    entry is taken: with one trusted proxy in front, that is the address the
    proxy itself observed, while the leftmost is whatever the client claimed.
    """
    if settings.RATE_LIMIT_TRUST_FORWARDED_FOR:
        forwarded = request.headers.get("X-Forwarded-For", "")
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    # request.client is None under some ASGI transports (notably in-process
    # test clients). Treat that as a single shared identity rather than
    # crashing or, worse, silently skipping the limit.
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(principal, request: Request, tenant_config=None) -> None:
    """
    Spends one token for this caller, or raises 429.

    Authenticated callers are keyed by their server-resolved `key_id`, which
    cannot be forged. Anonymous callers are keyed by peer address and given a
    smaller allowance, because their traffic is unattributable — there is no
    key to revoke if it turns out to be abusive.

    `tenant_config.rate_limit_rpm`, when set, overrides the tier default for
    authenticated callers — a tenant's SLA, not a per-key setting. It never
    applies to anonymous traffic: an unauthenticated caller has no verified
    tenant to carry an override from. Bucketing stays PER-KEY even when the
    rate comes from the tenant: two keys under one tenant sharing a bucket
    would let one integration's retry storm exhaust the other's budget, which
    defeats the reason key-level bucketing exists in the first place.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    if principal.authenticated:
        identity = f"key:{principal.key_id}"
        rpm = settings.RATE_LIMIT_AUTHENTICATED_RPM
        if tenant_config is not None and tenant_config.rate_limit_rpm is not None:
            rpm = tenant_config.rate_limit_rpm
    else:
        identity = f"ip:{_client_address(request)}"
        rpm = settings.RATE_LIMIT_ANONYMOUS_RPM

    capacity, refill = bucket_parameters(rpm, settings.RATE_LIMIT_BURST_SECONDS)
    allowed, retry_after = assess_rate_limiter.check(identity, capacity, refill)

    if not allowed:
        # Log the identity, never the prompt: a rate-limit event is an
        # operational signal and must not become a second copy of user content.
        logger.warning(f"Rate limit exceeded for {identity} (limit {rpm}/min).")
        metrics.rate_limited_total.labels(
            authenticated=str(principal.authenticated).lower()
        ).inc()
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit is {rpm:g} requests per minute.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )


def _reject_suspended_and_rate_limit(principal, request: Request) -> None:
    """
    Shared tail for every endpoint gated on a resolved principal (Phase 8
    hardening): reject a suspended tenant, then spend one rate-limit
    token. The original four endpoints (assess/assess_output/gateway_chat/
    tools_call) inline this same sequence; every endpoint gaining it now
    -- whoami, activity/trace/logs, gateway/tools/benchmarks catalogues,
    the policy editor, and the review endpoints -- previously had NEITHER
    check at all: a suspended tenant could poll them indefinitely, and
    several do real work per call (a 20MB-bounded audit-log scan, a
    policy-file write-then-validate-then-delete) that a caller with no
    throttling could repeat as fast as the network allows, a genuine
    resource-exhaustion gap even though none of them expose the kind of
    secret a rate limit would meaningfully protect against guessing --
    API keys here are 256-bit random tokens (`secrets.token_urlsafe(32)`),
    for which online brute-forcing is not a realistic threat regardless
    of throttling.
    """
    tenant_config = resolve_tenant(principal.tenant)
    if tenant_config.suspended:
        raise HTTPException(status_code=403, detail=f"Tenant '{principal.tenant}' is suspended.")
    _enforce_rate_limit(principal, request, tenant_config)


@app.on_event("startup")
async def warm_models() -> None:
    """
    Loads the models before the first request rather than during it.

    WHY THIS IS A CORRECTNESS FIX, not a performance tweak: cold loading of
    the embedding model plus the three fusion detectors measured ~35s on the
    reference machine, which exceeds ASSESS_TIMEOUT_SECONDS. Lazily loaded,
    the first request after every deploy would therefore be guaranteed to hit
    the deadline and return 503 — and so would every request queued behind it.
    Introducing a timeout without this would have converted a slow first
    request into a broken one.

    Failures are logged, not raised. A model that cannot load is a real
    problem, but the existing pipeline already degrades safely when a detector
    is unavailable (core/fusion.py never imputes a missing score), and /health
    already reports per-dependency status. Refusing to boot would turn a
    degraded gateway into no gateway.
    """
    if not settings.WARM_MODELS_ON_STARTUP:
        logger.info("Model warm-up disabled; models will load on first request.")
        return

    def _warm():
        from core.embeddings import _get_model
        from core.fusion import warm_up
        from core.threat_centroid import _get_malicious_centroid

        _get_model()                 # the sentence encoder, used by every stage
        _get_malicious_centroid()    # anchor centroid; embeds 15 anchors on
                                     # first use, which measured ~7s of the
                                     # first request even after the detectors
                                     # were already warm
        return warm_up()             # policy + the live fusion detectors

    started = time.perf_counter()
    try:
        warmed, detail = await asyncio.get_running_loop().run_in_executor(
            _assess_pool, _warm
        )
        elapsed = time.perf_counter() - started
        if warmed:
            logger.info(f"Model warm-up complete in {elapsed:.1f}s — {detail}")
        else:
            logger.warning(f"Model warm-up incomplete after {elapsed:.1f}s — {detail}")
    except Exception:
        logger.exception("Model warm-up failed; models will load on first request.")

    if settings.REGISTER_DEMO_TOOLS:
        from core.demo_tools import register_demo_tools
        register_demo_tools()
        logger.info("Demo tools registered (REGISTER_DEMO_TOOLS=true) — "
                   "POST /api/v1/tools/call can reach demo.* tools.")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()

    # Resolve the correlation ID BEFORE handling, so the handler and the audit
    # record it writes can both see it.
    request_id = _resolve_request_id(request)
    request.state.request_id = request_id

    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = str(process_time)
    # Echo the ID so the caller can correlate its own logs with ours, and so a
    # client that did not send one still learns what to quote in a bug report.
    response.headers[settings.REQUEST_ID_HEADER] = request_id

    if settings.METRICS_ENABLED:
        metrics.request_duration_seconds.labels(
            endpoint=_route_template(request),
            method=request.method,
            status=str(response.status_code),
        ).observe(process_time)

    return response


@app.get(settings.METRICS_PATH, include_in_schema=False)
def metrics_endpoint(request: Request):
    """
    Prometheus exposition.

    Deliberately excluded from the OpenAPI schema: it is an operational
    surface, not part of the API contract, and advertising it in public docs
    invites scraping by things that are not the monitoring system.
    """
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics are disabled.")

    if settings.METRICS_REQUIRE_AUTH:
        principal = resolve_principal(authorization=request.headers.get("Authorization"))
        if not principal.authenticated:
            raise HTTPException(
                status_code=401,
                detail="Metrics require authentication.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Sampled at scrape time rather than tracked on transition — see the
    # rationale in core/metrics.py.
    metrics.refresh_circuit_breaker_gauges()

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/assess", response_model=AssessResponse)
async def assess_prompt(req: AssessRequest, request: Request, background_tasks: BackgroundTasks):
    request.state.start_time = time.perf_counter()

    # 0. AUTHENTICATION — the security boundary.
    #    Capability comes from a verified credential, never from the request
    #    body. An anonymous caller gets GENERAL (least privilege); it cannot
    #    escalate by asserting anything about itself.
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 0b. TENANT RESOLVER — identity and SLA only, never policy (see
    #     core/tenancy.py). Placed right after auth and before rate limiting
    #     on purpose: a suspended tenant is rejected before spending a rate
    #     token or reaching detection, and the SLA it carries (rate_limit_rpm)
    #     must be known before the limiter runs.
    tenant_config = resolve_tenant(principal.tenant)
    if tenant_config.suspended:
        logger.warning(f"Rejected request for suspended tenant '{principal.tenant}'.")
        raise HTTPException(
            status_code=403,
            detail=f"Tenant '{principal.tenant}' is suspended.",
        )

    # 0c. RATE LIMITING — before any expensive work is scheduled. Placing this
    #     after authentication is deliberate: the limit that applies depends on
    #     whether the caller has a verified identity.
    _enforce_rate_limit(principal, request, tenant_config)

    logger.info(
        f"Assess request | capability={principal.capability} "
        f"tenant={principal.tenant} key_id={principal.key_id} "
        f"authenticated={principal.authenticated}"
    )

    # 1. Privacy Layer
    clean_query, redacted_info = redact_pii(req.prompt)
    redacted_items = redacted_info.get("items", [])

    # 2. Risk Assessment (Neuro-Symbolic) - offloaded to thread to avoid blocking event loop.
    #    background_tasks.add_task is passed through as the async scheduler:
    #    if Stage 4 needs judge arbitration, the caller gets the fast Ollama
    #    verdict now, and Llama Guard's slower, more accurate confirmation
    #    runs after this request completes (see
    #    core.risk.llama_guard_async_confirmation) rather than adding its
    #    multi-second tail latency to this response.
    try:
        risk_level, details = await _run_bounded(
            assess_risk, clean_query, background_tasks.add_task, req.prompt
        )
    except asyncio.TimeoutError:
        # 503, NOT a fabricated BLOCK verdict. A timeout is an availability
        # event, not a security finding, and writing a verdict no analysis
        # produced would put a lie in the audit log — the exact failure mode
        # §2 of the engineering assessment warns about, where infrastructure
        # failure masquerades as detection signal. The integration contract is
        # "anything other than 200 means do not proceed", which keeps the
        # caller fail-closed without corrupting the record of why.
        logger.error(
            f"Assessment timed out after {settings.ASSESS_TIMEOUT_SECONDS}s "
            f"| key_id={principal.key_id} request_id={request.state.request_id}"
        )
        metrics.assessment_timeouts_total.labels(endpoint="/api/v1/assess").inc()
        raise HTTPException(
            status_code=503,
            detail=(
                f"Assessment did not complete within "
                f"{settings.ASSESS_TIMEOUT_SECONDS:g}s. The prompt was NOT "
                f"assessed and must not be treated as approved."
            ),
            headers={"Retry-After": "5"},
        )

    # 3. Policy Context / Arbitration — using the SERVER-RESOLVED capability
    #    AND the SERVER-RESOLVED tenant (never a client-asserted one, same
    #    boundary auth.py already enforces for capability). A tenant's
    #    policy is looked up independently of the Tenant Resolver step
    #    above — they are separate concerns (identity/SLA vs. enforcement
    #    mapping; see core/policy.py's module docstring) that happen to be
    #    keyed by the same tenant_id.
    decision, reason = policy_decision(principal.capability, risk_level, principal.tenant)

    # 3b. OUTPUT GUARD, IN THE SAME CALL — closes the V2 Phase 0 gap where
    #     Output Guard was a separate endpoint a caller had to remember to
    #     invoke correctly after generating a response. Only runs when the
    #     caller submitted response_text AND the input wasn't already
    #     BLOCKed or sent to REVIEW — checking the output of a prompt that
    #     was never allowed through in the first place (or isn't allowed
    #     through YET, pending a human) answers a question nobody asked.
    #
    #     The final decision is the MORE SEVERE of the two
    #     (BLOCK > REVIEW > RESTRICT > ALLOW): a clean prompt with a leaky
    #     or toxic response must still BLOCK — that is the entire point of
    #     checking output at all, not an edge case of it.
    output_decision = None
    output_details = None
    if req.response_text is not None and decision not in ("BLOCK", "REVIEW"):
        from core.output_guardrails import assess_output

        try:
            output_decision, output_details = await _run_bounded(
                assess_output, req.response_text, req.system_prompt
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Output assessment (combined call) timed out after "
                f"{settings.ASSESS_TIMEOUT_SECONDS}s | key_id={principal.key_id} "
                f"request_id={request.state.request_id}"
            )
            metrics.assessment_timeouts_total.labels(endpoint="/api/v1/assess (output)").inc()
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Output assessment did not complete within "
                    f"{settings.ASSESS_TIMEOUT_SECONDS:g}s. Neither the prompt "
                    f"nor the response were assessed and must not be treated "
                    f"as approved."
                ),
                headers={"Retry-After": "5"},
            )

        _SEVERITY = {"ALLOW": 0, "RESTRICT": 1, "REVIEW": 2, "BLOCK": 3}
        if _SEVERITY.get(output_decision, 3) > _SEVERITY.get(decision, 0):
            decision = output_decision

    # Update details with reason and the identity the decision was made under.
    details["policy_reason"] = reason
    details["principal"] = principal.to_audit()
    details["request_id"] = request.state.request_id
    details["tenant"] = principal.tenant
    if output_details is not None:
        details["output_assessment"] = output_details

    # 3c. HUMAN REVIEW (Phase 4) — a REVIEW decision is neither an ALLOW nor
    #     a BLOCK: it is queued for a human to resolve. No raw prompt text
    #     is stored in the queue, only its hash — see core/review_queue.py's
    #     docstring for why. The caller gets `review_id` back and polls
    #     GET /api/v1/review/{review_id} for the eventual outcome.
    review_id = None
    if decision == "REVIEW":
        review = enqueue_review(
            reason=reason, capability=principal.capability, risk=risk_level,
            tenant=principal.tenant, prompt_hash=hashlib.sha256(clean_query.encode("utf-8")).hexdigest(),
            request_id=request.state.request_id,
        )
        review_id = review.review_id
        details["review_id"] = review_id

    # 4. Audit Logging — the FINAL combined decision, not the pre-output-check
    #    input decision, so a record never shows "ALLOW" for a request that
    #    was actually blocked on its response.
    log_event(principal.capability, clean_query, risk_level, decision, details)

    # 4b. A DISTINCT output audit event, in addition to the merged summary
    #     above (details["output_assessment"]) — see log_output_event's
    #     docstring for why an output decision needs its own record rather
    #     than living only nested inside the input's.
    if output_details is not None:
        log_output_event(
            principal.capability, req.response_text, output_decision,
            output_details, tenant=principal.tenant, request_id=request.state.request_id,
        )

    # 5. Metrics. After the audit write, deliberately: the audit record is the
    #    compliance artefact and must not be at risk from an instrumentation
    #    bug. Wrapped for the same reason — observability must never be able to
    #    fail a request that was otherwise assessed and decided successfully.
    if settings.METRICS_ENABLED:
        try:
            metrics.record_assessment(decision, risk_level, details, principal.tenant)
        except Exception:
            logger.exception("Failed to record assessment metrics; serving anyway.")

    # Calculate process time explicitly for the response body (if needed by client directly)
    # The header is already set by middleware, but this makes it visible in the JSON
    # We'll just pass 0.0 here since the middleware calculates the true end-to-end,
    # but the client can read the header. Alternatively, we can calculate it here:
    process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2) if hasattr(request.state, "start_time") else 0.0

    return AssessResponse(
        decision=decision,
        risk_level=risk_level,
        topicality=details.get("topicality", "UNKNOWN"),
        capability=principal.capability,
        authenticated=principal.authenticated,
        details=details,
        clean_prompt=clean_query,
        redacted_items=redacted_items,
        process_time_ms=process_time_ms,
        output_decision=output_decision,
        output_details=output_details,
        clean_response=output_details.get("clean_response") if output_details else None,
        review_id=review_id,
    )

@app.post("/api/v1/assess_output", response_model=AssessOutputResponse)
async def assess_output_endpoint(req: AssessOutputRequest, request: Request):
    request.state.start_time = time.perf_counter()

    # This endpoint previously had NO authentication and NO limiting, which
    # made every control on /assess optional: an attacker wanting to exhaust
    # the worker pool would simply use this one instead. It runs the same
    # expensive machinery, so it gets the same gate. Enforcing AUTH_MODE here
    # too closes a real gap — "required" is documented as meaning every
    # request is attributed, and one open expensive endpoint made that false.
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Same reasoning as /assess: a suspended tenant must not have a working
    # bypass route just because it happens to be the OTHER endpoint.
    tenant_config = resolve_tenant(principal.tenant)
    if tenant_config.suspended:
        logger.warning(f"Rejected output-assessment request for suspended tenant '{principal.tenant}'.")
        raise HTTPException(
            status_code=403,
            detail=f"Tenant '{principal.tenant}' is suspended.",
        )

    _enforce_rate_limit(principal, request, tenant_config)

    logger.info(f"Output assessment request | key_id={principal.key_id} tenant={principal.tenant}")

    from core.output_guardrails import assess_output

    # 1. Output Assessment (secrets, system-prompt leakage, PII, toxicity, hallucination)
    try:
        decision, details = await _run_bounded(assess_output, req.response_text, req.system_prompt)
    except asyncio.TimeoutError:
        logger.error(
            f"Output assessment timed out after "
            f"{settings.ASSESS_TIMEOUT_SECONDS}s | key_id={principal.key_id} "
            f"request_id={request.state.request_id}"
        )
        metrics.assessment_timeouts_total.labels(endpoint="/api/v1/assess_output").inc()
        raise HTTPException(
            status_code=503,
            detail=(
                f"Output assessment did not complete within "
                f"{settings.ASSESS_TIMEOUT_SECONDS:g}s. The response was NOT "
                f"assessed and must not be treated as approved."
            ),
            headers={"Retry-After": "5"},
        )

    # 2. Audit Logging — previously MISSING entirely on this endpoint, so a
    #    BLOCK here (PII leak, secret, toxicity, hallucination) left zero
    #    audit trail. See log_output_event's docstring.
    log_output_event(
        principal.capability, req.response_text, decision, details,
        tenant=principal.tenant, request_id=request.state.request_id,
    )

    process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2) if hasattr(request.state, "start_time") else 0.0

    return AssessOutputResponse(
        decision=decision,
        details=details,
        process_time_ms=process_time_ms,
        clean_response=details.get("clean_response"),
    )


@app.get("/api/v1/gateway/providers")
def gateway_providers(request: Request):
    """
    Developer UI's Model Gateway view: which provider TYPES this
    deployment can actually route a `/api/v1/gateway/chat` call to
    (`core.llm_providers.list_provider_names`, the real registry backing
    `get_provider`), and which one is the default when a caller omits
    `provider`. No credentials or endpoint URLs are returned -- those
    live in environment/settings, not in an HTTP response.

    INTERNAL capability required, matching every other Developer UI
    endpoint added alongside this one (tools listing, raw logs) -- this
    is operator/developer-facing configuration, not end-user data.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to view gateway configuration.",
        )
    _reject_suspended_and_rate_limit(principal, request)
    return {
        "providers": list_provider_names(),
        "default_provider": settings.LLM_GATEWAY_DEFAULT_PROVIDER,
    }


@app.get("/api/v1/tools")
def list_tools(request: Request):
    """
    Developer UI's Tool Gateway view: every tool currently registered
    with the shared ToolRegistry (`core.tools.get_tool_registry`) --
    name, description, JSON-Schema parameters, risk level, and required
    capability. This is the tool CATALOGUE (what could be called and
    under what rules), not a log of calls -- see /api/v1/activity or
    /api/v1/logs (event_type=tool_call) for actual call history.

    INTERNAL capability required, same bar as the other new Developer UI
    endpoints -- a tool's registered schema can reveal internal system
    shape (table names, internal service paths) that a GENERAL/ELEVATED
    caller has no operational need to browse.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to view the tool catalogue.",
        )
    _reject_suspended_and_rate_limit(principal, request)
    registry = get_tool_registry()
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "risk_level": spec.risk_level,
                "capability_required": spec.capability_required,
            }
            for spec in registry.list_tools()
        ]
    }


@app.post("/api/v1/gateway/chat", response_model=GatewayChatResponse)
async def gateway_chat(req: GatewayChatRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Real LLM Gateway (Phase 5) — request forwarding + response
    interception. Unlike /api/v1/assess (a sidecar the caller submits an
    ALREADY-GENERATED response to), this endpoint decides whether the call
    to the real LLM happens AT ALL: input guardrails run first, and the
    provider is only ever called if they allow it.

    Order: input assessment -> policy decision -> (BLOCK/REVIEW stops here,
    no provider call) -> forward to the provider -> output assessment on
    what it returned -> final combined decision, same severity ordering as
    /api/v1/assess (BLOCK > REVIEW > RESTRICT > ALLOW).
    """
    request.state.start_time = time.perf_counter()

    # 0. AUTH + TENANT + RATE LIMIT — identical boundary to /api/v1/assess.
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    tenant_config = resolve_tenant(principal.tenant)
    if tenant_config.suspended:
        raise HTTPException(status_code=403, detail=f"Tenant '{principal.tenant}' is suspended.")
    _enforce_rate_limit(principal, request, tenant_config)

    # 0b. TOKEN QUOTA (Phase 5: token accounting) — checked before the
    # (comparatively expensive) input assessment runs at all, same
    # cheapest-check-first ordering as rate limiting above. Enforced against
    # usage ALREADY recorded from past calls, not this one — see
    # core/token_quota.py's module docstring for why a pre-flight cost
    # estimate isn't possible here.
    if settings.GATEWAY_TOKEN_QUOTA_ENABLED:
        quota = (
            tenant_config.token_quota_daily
            if tenant_config.token_quota_daily is not None
            else settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT
        )
        if gateway_token_quota.would_exceed(principal.tenant, quota):
            if settings.METRICS_ENABLED:
                metrics.gateway_quota_rejections_total.labels(tenant=principal.tenant).inc()
            raise HTTPException(
                status_code=429,
                detail=f"Tenant {principal.tenant!r} has exhausted its daily token "
                       f"quota ({quota}). The provider was NOT called.",
                headers={"Retry-After": str(int(seconds_until_utc_midnight()))},
            )

    # 1. INPUT GUARDRAILS — before any external call. This is the actual
    #    architectural difference from a sidecar: Gatekeeper decides
    #    whether the provider is called at all, not just whether to trust
    #    a response the caller already obtained.
    clean_query, redacted_info = redact_pii(req.prompt)
    try:
        risk_level, details = await _run_bounded(assess_risk, clean_query, background_tasks.add_task, req.prompt)
    except asyncio.TimeoutError:
        metrics.assessment_timeouts_total.labels(endpoint="/api/v1/gateway/chat (input)").inc()
        raise HTTPException(
            status_code=503,
            detail=f"Input assessment did not complete within {settings.ASSESS_TIMEOUT_SECONDS:g}s. "
                   f"The provider was NOT called.",
            headers={"Retry-After": "5"},
        )

    decision, reason = policy_decision(principal.capability, risk_level, principal.tenant)
    details["policy_reason"] = reason
    details["principal"] = principal.to_audit()
    details["request_id"] = request.state.request_id
    details["tenant"] = principal.tenant

    review_id = None
    if decision == "REVIEW":
        review = enqueue_review(
            reason=reason, capability=principal.capability, risk=risk_level,
            tenant=principal.tenant, prompt_hash=hashlib.sha256(clean_query.encode("utf-8")).hexdigest(),
            request_id=request.state.request_id,
        )
        review_id = review.review_id
        details["review_id"] = review_id

    log_event(principal.capability, clean_query, risk_level, decision, details)

    # BLOCK/REVIEW stop here — the provider is never called. Same reasoning
    # as /api/v1/assess skipping output assessment for these two: a prompt
    # that isn't allowed through (yet, or at all) has no response to forward.
    if decision in ("BLOCK", "REVIEW"):
        if settings.METRICS_ENABLED:
            metrics.gateway_call_total.labels(
                provider=req.provider or settings.LLM_GATEWAY_DEFAULT_PROVIDER,
                outcome="blocked" if decision == "BLOCK" else "review",
            ).inc()
        process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2)
        return GatewayChatResponse(
            decision=decision, content=None, provider=None, model=None,
            usage=None, review_id=review_id, details=details, process_time_ms=process_time_ms,
        )

    # 2. FORWARD TO THE PROVIDER, with fallback across a configured chain
    #    (Phase 5: "Timeout/failure handling, fallback"). Fallback only
    #    applies when the caller left `provider` unset — an explicit choice
    #    is never second-guessed, see GATEWAY_FALLBACK_PROVIDERS's docstring
    #    in core/config.py.
    provider_name = req.provider or settings.LLM_GATEWAY_DEFAULT_PROVIDER
    fallback_chain = []
    if req.provider is None and settings.GATEWAY_FALLBACK_PROVIDERS:
        fallback_chain = [
            p.strip() for p in settings.GATEWAY_FALLBACK_PROVIDERS.split(",")
            if p.strip() and p.strip() != provider_name
        ]
    attempt_names = [provider_name] + fallback_chain

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": clean_query})

    llm_response = None
    used_provider_name = None
    last_error_status = None
    last_error_detail = None

    for attempt_index, attempt_name in enumerate(attempt_names):
        is_primary = attempt_index == 0
        try:
            provider = get_provider(attempt_name)
        except KeyError:
            if is_primary:
                raise HTTPException(status_code=422, detail=f"Unknown provider: {attempt_name!r}")
            logger.error(f"Configured fallback provider {attempt_name!r} is unknown; skipping it.")
            continue

        # A model name the caller gave is meant for the PRIMARY provider —
        # forwarding it to a different fallback provider would silently
        # request a model that provider may not have. Fallback attempts
        # always use that provider's own configured default instead.
        attempt_model = req.model if is_primary else None

        call_start = time.perf_counter()
        try:
            llm_response = await _run_gateway_bounded(provider.complete, messages, attempt_model)
            used_provider_name = attempt_name
            latency_ms = round((time.perf_counter() - call_start) * 1000, 2)
            break
        except asyncio.TimeoutError:
            latency_ms = round((time.perf_counter() - call_start) * 1000, 2)
            log_gateway_event(principal.capability, attempt_name, attempt_model, success=False,
                              latency_ms=latency_ms, decision="BLOCK", error="timeout",
                              tenant=principal.tenant, request_id=request.state.request_id)
            if settings.METRICS_ENABLED:
                metrics.gateway_call_total.labels(provider=attempt_name, outcome="timeout").inc()
            last_error_status = 503
            last_error_detail = (f"The LLM provider did not respond within "
                                 f"{settings.GATEWAY_TIMEOUT_SECONDS:g}s.")
        except LLMProviderError as e:
            latency_ms = round((time.perf_counter() - call_start) * 1000, 2)
            log_gateway_event(principal.capability, attempt_name, attempt_model, success=False,
                              latency_ms=latency_ms, decision="BLOCK", error=str(e),
                              tenant=principal.tenant, request_id=request.state.request_id)
            if settings.METRICS_ENABLED:
                metrics.gateway_call_total.labels(provider=attempt_name, outcome="failure").inc()
            last_error_status = 502
            last_error_detail = f"The LLM provider call failed: {e}"

    if llm_response is None:
        # Every attempt in the chain (primary + fallbacks) failed or timed
        # out — raise using the LAST attempt's error, the one closest to
        # what actually stopped the request.
        raise HTTPException(
            status_code=last_error_status,
            detail=last_error_detail,
            headers={"Retry-After": "5"} if last_error_status == 503 else None,
        )

    provider_name = used_provider_name
    if provider_name != (req.provider or settings.LLM_GATEWAY_DEFAULT_PROVIDER):
        details["gateway_fallback_used"] = True
        details["gateway_fallback_from"] = req.provider or settings.LLM_GATEWAY_DEFAULT_PROVIDER

    # 3. OUTPUT GUARDRAILS — same machinery as /api/v1/assess's combined path.
    from core.output_guardrails import assess_output
    try:
        output_decision, output_details = await _run_bounded(
            assess_output, llm_response.content, req.system_prompt
        )
    except asyncio.TimeoutError:
        metrics.assessment_timeouts_total.labels(endpoint="/api/v1/gateway/chat (output)").inc()
        raise HTTPException(
            status_code=503,
            detail=f"Output assessment did not complete within {settings.ASSESS_TIMEOUT_SECONDS:g}s. "
                   f"The provider responded but its output was NOT assessed and must not be shown.",
            headers={"Retry-After": "5"},
        )

    _SEVERITY = {"ALLOW": 0, "RESTRICT": 1, "REVIEW": 2, "BLOCK": 3}
    final_decision = decision
    if _SEVERITY.get(output_decision, 3) > _SEVERITY.get(final_decision, 0):
        final_decision = output_decision

    details["output_assessment"] = output_details
    log_output_event(principal.capability, llm_response.content, output_decision, output_details,
                     tenant=principal.tenant, request_id=request.state.request_id)
    log_gateway_event(principal.capability, provider_name, llm_response.model, success=True,
                      latency_ms=latency_ms, decision=final_decision, usage=llm_response.usage,
                      tenant=principal.tenant, request_id=request.state.request_id)
    if settings.METRICS_ENABLED:
        metrics.gateway_call_total.labels(provider=provider_name, outcome="success").inc()

    # Token accounting (Phase 5) — recorded AFTER the call, since usage is
    # only known once the provider has answered. See core/token_quota.py's
    # module docstring for why this can only ever gate the NEXT call.
    tokens_used = extract_total_tokens(llm_response.usage)
    if tokens_used > 0:
        gateway_token_quota.record(principal.tenant, tokens_used)
        if settings.METRICS_ENABLED:
            metrics.gateway_tokens_total.labels(tenant=principal.tenant).inc(tokens_used)

    process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2)
    return GatewayChatResponse(
        decision=final_decision,
        content=output_details.get("clean_response") if final_decision != "BLOCK" else None,
        provider=llm_response.provider,
        model=llm_response.model,
        usage=llm_response.usage,
        review_id=None,
        details=details,
        process_time_ms=process_time_ms,
    )


@app.post("/api/v1/tools/call", response_model=ToolCallResponse)
async def tools_call(req: ToolCallRequest, request: Request):
    """
    Real LLM Gateway's tool-calling counterpart (Phase 6, Tool/Agent
    Gateway) — the endpoint `core/tools.py::execute_tool` existed without
    since the item that built it, wired up now that there is a caller to
    give it real `tenant`/`request_id` context and a real REVIEW path.

    Same auth/tenant/rate-limit boundary as every other endpoint. The
    actual decision — access control, structural validation, risk-based
    approval, execution, audit — is entirely `execute_tool`'s; this
    endpoint's own job is the HTTP contract and, for a REVIEW decision,
    enqueuing it so a human can actually resolve it (previously
    documented as deferred in `execute_tool`'s own docstring: "no real
    tool call path to enforce this against" — there is one now).
    """
    request.state.start_time = time.perf_counter()

    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    tenant_config = resolve_tenant(principal.tenant)
    if tenant_config.suspended:
        raise HTTPException(status_code=403, detail=f"Tenant '{principal.tenant}' is suspended.")
    _enforce_rate_limit(principal, request, tenant_config)

    result = execute_tool(
        principal.capability, req.name, req.arguments,
        tenant=principal.tenant, request_id=request.state.request_id,
    )

    review_id = None
    if result["decision"] == "REVIEW":
        # Same identifying-hash-not-raw-content contract ReviewRecord's
        # prompt_hash already promises (core/review_queue.py) — a tool
        # call's name+arguments hash fits it exactly, even though the
        # field name is inherited from Phase 4's prompt-oriented origin.
        call_hash = hashlib.sha256(
            json.dumps({"tool": req.name, "arguments": req.arguments},
                      sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        review = enqueue_review(
            reason=result["reason"], capability=principal.capability,
            risk=result.get("risk_level") or "HIGH", tenant=principal.tenant,
            prompt_hash=call_hash, request_id=request.state.request_id,
        )
        review_id = review.review_id

    process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2)
    return ToolCallResponse(
        decision=result["decision"],
        tool=result["tool"],
        reason=result["reason"],
        output=result.get("output"),
        error=result.get("error"),
        review_id=review_id,
        process_time_ms=process_time_ms,
    )


@app.post("/api/v1/cache/flush")
def flush_semantic_cache(request: Request):
    """
    INTERNAL capability required — same bar as the review endpoints above.
    Previously had no check at all (issue #5): any unauthenticated caller
    could repeatedly flush the semantic cache, forcing every subsequent
    request to pay the uncached detection cost again — a latency/cost
    amplification vector, not just an inconvenience.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to flush the semantic cache.",
        )
    _reject_suspended_and_rate_limit(principal, request)
    flush_cache()
    return {"status": "success"}


# --- Identity check (Phase 7) ---

@app.get("/api/v1/whoami", response_model=WhoAmIResponse)
def whoami(request: Request):
    """
    Resolves the caller's own credential against the real KeyStore and
    returns what it grants. Exists for the client UI's login page: a key
    typed into a browser is worth nothing until it is checked against the
    same resolve_principal() every enforcement decision uses -- this
    endpoint IS that check, not a separate, potentially-divergent one.

    Always requires a valid credential regardless of AUTH_MODE -- a login
    endpoint that accepts "no credential" as a pass would defeat the point
    of having one.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _reject_suspended_and_rate_limit(principal, request)
    return WhoAmIResponse(
        capability=principal.capability, tenant=principal.tenant, key_id=principal.key_id,
    )


# --- Client UI settings: privacy & protection (Phase 7) ---

@app.get("/api/v1/settings/privacy")
def privacy_settings(request: Request):
    """
    What PII protection is actually applied to every request -- the REAL
    running configuration from core/privacy.py (`REGEX_PATTERNS`'
    category names, `NER_LABELS`), not a description written by hand that
    could drift from what the code actually does. Global: this pipeline
    has no per-tenant privacy configuration today, so every caller sees
    the same answer -- an honest reflection of that, not a per-tenant
    settings page for a knob that doesn't exist yet.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _reject_suspended_and_rate_limit(principal, request)
    return {
        "regex_categories": sorted(REGEX_PATTERNS.keys()),
        "ner_labels": sorted(NER_LABELS),
    }


@app.get("/api/v1/settings/protection")
def protection_settings(request: Request):
    """
    What actually governs the caller's own requests: their tenant's real
    SLA config (`core.tenancy.resolve_tenant` -- status, rate limit,
    token quota) and the real risk->action mapping their OWN capability
    resolves to under their tenant's policy (`core.policy.
    resolve_policy_set`) -- the exact same lookups `/api/v1/assess` and
    friends make to decide ALLOW/BLOCK/RESTRICT/REVIEW, not a summary
    that could drift from what's actually enforced.

    Every caller sees their OWN tenant by default; INTERNAL may pass
    `?tenant=<other>` to inspect a different tenant's settings, same
    cross-tenant rule as the activity feed. The policy mapping shown is
    always for the REQUESTING principal's own capability -- "what
    governs me", not an arbitrary capability picker (the Policy Editor
    is where the full per-capability mapping for every tenant lives).
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _reject_suspended_and_rate_limit(principal, request)

    requested_tenant = request.query_params.get("tenant")
    tenant_id = requested_tenant if (principal.capability == CAPABILITY_INTERNAL and requested_tenant) else principal.tenant

    tenant_config = resolve_tenant(tenant_id)
    policy_set = resolve_policy_set(tenant_id)

    return {
        "tenant": {
            "tenant_id": tenant_config.tenant_id,
            "display_name": tenant_config.display_name,
            "status": tenant_config.status,
            "rate_limit_rpm": tenant_config.rate_limit_rpm,
            "token_quota_daily": tenant_config.token_quota_daily,
        },
        "policy": {
            "capability": principal.capability,
            "risk_to_action": policy_set.policies.get(principal.capability, {}),
            "default_action": get_default_action(),
        },
    }


# --- Activity feed (Phase 7) ---

def _resolve_activity_tenant_scope(principal, request: Request) -> str:
    """
    Shared by every audit-log-reading endpoint (activity feed, trace,
    logs): every caller defaults to their OWN tenant; only INTERNAL may
    pass `?tenant=<other>` or `?tenant=__all__` to cross tenants. Kept as
    one function so the three endpoints can never silently diverge on
    this rule.
    """
    requested_tenant = request.query_params.get("tenant")
    if principal.capability == CAPABILITY_INTERNAL and requested_tenant:
        return None if requested_tenant == "__all__" else requested_tenant
    return principal.tenant


@app.get("/api/v1/activity")
def activity_feed(request: Request, limit: int = 50, event_type: str = None):
    """
    Recent audit-log events for the client UI's activity feed. Reads the
    same audit.jsonl every governance decision is already written to
    (core/logger.py) -- this endpoint is a read view over it, not a new
    source of truth.

    Every authenticated caller may see their OWN tenant's activity
    (`principal.tenant`, never overridable by the caller) -- an operator's
    "what happened to my org's requests" is not the sensitive question the
    review-queue endpoints are gated on. INTERNAL is the one capability
    that may additionally cross tenants, matching list_reviews' own
    reasoning: seeing OTHER tenants' activity is exactly what a
    non-INTERNAL caller has no business doing. INTERNAL requests everyone's
    activity by passing tenant=__all__; anything else is still scoped to
    that one tenant, same as a non-INTERNAL caller.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _reject_suspended_and_rate_limit(principal, request)

    tenant_filter = _resolve_activity_tenant_scope(principal, request)
    event_types = [t.strip() for t in event_type.split(",")] if event_type else None
    return get_recent_activity(limit=limit, tenant=tenant_filter, event_types=event_types)


@app.get("/api/v1/activity/trace/{request_id}")
def activity_trace(request_id: str, request: Request):
    """
    Every audit event sharing one request_id, oldest first -- "what
    actually happened, end to end, for this one call": the input
    assessment's detector signals (semantic_score, symbolic_triggered,
    judge_invoked, risk), the policy decision, and (if the request went
    further) the gateway call or tool call it triggered, and the output
    assessment on the way back.

    Same tenant-scoping as the activity feed above: a non-INTERNAL caller
    only ever sees a trace if it belongs to their own tenant (an event
    list that's simply empty, not a 403 -- a stranger's request_id gives
    no signal either way about whether it exists).
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _reject_suspended_and_rate_limit(principal, request)

    tenant_filter = _resolve_activity_tenant_scope(principal, request)
    return find_by_request_id(request_id, tenant=tenant_filter)


@app.get("/api/v1/logs")
def raw_logs(request: Request, limit: int = 100, event_type: str = None):
    """
    The same audit log the activity feed reads, but for developers/
    operators: cross-tenant by default (every tenant at once) and a
    higher default limit, where the activity feed defaults to "my own
    tenant" for a general caller. INTERNAL capability required outright
    -- unlike the activity feed, there is no "your own tenant" reading of
    this endpoint to fall back to.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to view the raw cross-tenant log.",
        )
    _reject_suspended_and_rate_limit(principal, request)

    requested_tenant = request.query_params.get("tenant")
    tenant_filter = None if not requested_tenant or requested_tenant == "__all__" else requested_tenant
    event_types = [t.strip() for t in event_type.split(",")] if event_type else None
    return get_recent_activity(limit=limit, tenant=tenant_filter, event_types=event_types)


@app.get("/api/v1/benchmarks")
def benchmarks(request: Request):
    """
    Developer UI's Benchmarks view: this project's own real benchmark run
    results (`core.benchmarks.list_benchmark_runs`, reading
    `_evidence/benchmark_results_*.json` off disk) -- the exact reports
    this project's benchmark scripts produce and commit as evidence, not
    a re-derived or re-run summary.

    INTERNAL capability required, same bar as the other Developer UI
    endpoints -- these numbers describe the deployment's own detection
    performance, not something a GENERAL/ELEVATED caller has an
    operational need to browse.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to view benchmark results.",
        )
    _reject_suspended_and_rate_limit(principal, request)
    return list_benchmark_runs()


# --- Policy Editor (Phase 7) ---
#
# Wraps the SAME functions scripts/manage_policy_versions.py already uses
# (validate_policy_file, deploy_policy, rollback_to) -- this is an HTTP
# front end for that real, already-safe machinery, not a new write path
# with its own logic. In particular deploy NEVER skips validation: see
# deploy_policy's own docstring for why that discipline lives at the
# caller, not inside it.

def _require_internal(request: Request):
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(status_code=403, detail="INTERNAL capability required for policy administration.")
    _reject_suspended_and_rate_limit(principal, request)
    return principal


@app.get("/api/v1/policy")
def get_policy(request: Request):
    """Current live policy (parsed) plus the real snapshot history a
    deploy could roll back to."""
    _require_internal(request)
    try:
        with open(settings.POLICY_RULES_FILE, "r", encoding="utf-8") as f:
            raw_text = f.read()
        content = load_policy_file()
    except FileNotFoundError:
        raw_text, content = None, None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live policy file is unreadable: {e}")

    return {
        "path": settings.POLICY_RULES_FILE,
        "raw_text": raw_text,
        "content": content,
        "versions": list_versions(),
    }


@app.post("/api/v1/policy/validate")
def validate_policy(req: PolicyContentRequest, request: Request):
    """Validates candidate policy content WITHOUT touching the live file
    -- writes it to a throwaway temp file (same extension as the live
    file, so YAML-vs-JSON dispatch matches), runs the real
    `validate_policy_file`, always cleans up."""
    _require_internal(request)
    suffix = os.path.splitext(settings.POLICY_RULES_FILE)[1] or ".json"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
        tmp.write(req.content)
        tmp_path = tmp.name
    try:
        errors = validate_policy_file(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"valid": not errors, "errors": errors}


@app.post("/api/v1/policy/deploy")
def deploy_policy_endpoint(req: PolicyContentRequest, request: Request):
    """Validates first and REFUSES to deploy on any error -- the exact
    discipline `scripts/manage_policy_versions.py::cmd_deploy` already
    applies, reproduced here rather than bypassed because the caller is
    now HTTP instead of a CLI operator. Snapshots the outgoing policy
    before overwriting it (deploy_policy's own behaviour) and reloads it
    into the live process immediately -- no restart, no window where the
    old and new policy could both be enforced depending on which worker
    handled a request."""
    principal = _require_internal(request)
    suffix = os.path.splitext(settings.POLICY_RULES_FILE)[1] or ".json"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
        tmp.write(req.content)
        tmp_path = tmp.name
    try:
        errors = validate_policy_file(tmp_path)
        if errors:
            metrics.policy_changes_total.labels(action="deploy", outcome="rejected").inc()
            raise HTTPException(status_code=422, detail={"message": "Refusing to deploy: invalid policy.", "errors": errors})
        previous_version = deploy_policy(tmp_path)
    finally:
        os.unlink(tmp_path)

    metrics.policy_changes_total.labels(action="deploy", outcome="success").inc()
    logger.warning(f"Policy deployed via API by key_id={principal.key_id}. Previous snapshot: {previous_version}")
    return {"deployed": True, "previous_version": previous_version}


@app.post("/api/v1/policy/rollback")
def rollback_policy_endpoint(req: PolicyRollbackRequest, request: Request):
    """Restores a named snapshot over the live policy and reloads it --
    itself snapshotted first by `rollback_to`, so a bad rollback is just
    as undoable as a bad deploy."""
    principal = _require_internal(request)
    try:
        rollback_to(req.version)
    except FileNotFoundError as e:
        metrics.policy_changes_total.labels(action="rollback", outcome="rejected").inc()
        raise HTTPException(status_code=404, detail=str(e))

    metrics.policy_changes_total.labels(action="rollback", outcome="success").inc()
    logger.warning(f"Policy rolled back via API by key_id={principal.key_id} to version={req.version}")
    return {"rolled_back_to": req.version}


# --- Human Review (Phase 4) ---

@app.get("/api/v1/review/{review_id}", response_model=ReviewStatusResponse)
def get_review_status(review_id: str, request: Request):
    """
    Status check for a REVIEW decision. Any authenticated caller may poll
    this — read access to "is my request still pending" is far less
    sensitive than the ability to approve/reject one, so it is gated only
    on authentication, not on capability (unlike the two endpoints below).
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if auth_required() and not principal.authenticated:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Present a valid API key as "
                   "'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _reject_suspended_and_rate_limit(principal, request)

    review = get_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=f"No such review: {review_id}")
    return ReviewStatusResponse(**review)


@app.get("/api/v1/review")
def list_reviews(request: Request):
    """
    Lists PENDING reviews only — a reviewer's queue, not a full history.
    Resolved reviews stay retrievable individually via the status-check
    endpoint above; this endpoint is for "what needs my attention right
    now", not an audit export (that is what audit.jsonl is for).

    INTERNAL capability required — this is where a caller could otherwise
    learn about OTHER tenants' pending requests, which GENERAL/ELEVATED
    callers have no business seeing.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to list pending reviews.",
        )
    _reject_suspended_and_rate_limit(principal, request)
    return {"pending": list_pending_reviews()}


@app.post("/api/v1/review/{review_id}/resolve", response_model=ReviewStatusResponse)
def resolve_review_endpoint(review_id: str, req: ReviewResolveRequest, request: Request):
    """
    Approve or reject a pending review — "feeding back into the policy
    engine" (Phase 4 roadmap item): the resolution becomes the request's
    FINAL decision (APPROVED -> ALLOW, REJECTED -> BLOCK), retrievable via
    the status-check endpoint above. Does NOT retroactively alter the
    semantic cache for future identical prompts — see core/review_queue.py's
    docstring for why (no raw prompt text is stored here to re-embed from,
    a deliberate privacy choice, not an oversight).

    INTERNAL capability required, same reasoning as listing above, and more
    so: this is a write that changes what request actually gets enforced.
    """
    principal = resolve_principal(authorization=request.headers.get("Authorization"))
    if principal.capability != CAPABILITY_INTERNAL:
        raise HTTPException(
            status_code=403,
            detail="INTERNAL capability required to resolve a review.",
        )
    _reject_suspended_and_rate_limit(principal, request)

    try:
        record = resolve_review(review_id, req.outcome, reviewer=principal.key_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No such review: {review_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return ReviewStatusResponse(**record)

@app.get("/health")
def health_check():
    import os
    import requests
    from core.config import POLICY_FILE, POLICY_RULES_FILE, OLLAMA_API_URL
    from core.privacy import NLP_MODEL
    from core.embeddings import _get_model
    
    status = {
        "status": "healthy",
        "checks": {
            "policy_files": False,
            "spacy_model": False,
            "embedding_model": False,
            "semantic_judge": False
        }
    }
    
    # 1. Check Policy Files
    if os.path.exists(POLICY_FILE) and os.path.exists(POLICY_RULES_FILE):
        status["checks"]["policy_files"] = True
    else:
        status["status"] = "degraded"
        
    # 2. Check spaCy Model
    if NLP_MODEL is not None:
        status["checks"]["spacy_model"] = True
    else:
        status["status"] = "degraded"
        
    # 3. Check Embedding Model
    try:
        model = _get_model()
        if model is not None:
            status["checks"]["embedding_model"] = True
    except Exception:
        status["status"] = "degraded"
        
    # 4. Check Semantic Judge (Ollama)
    #
    # Phase 8 hardening: a real load test (docs/ROADMAP_V2.md's Phase 8
    # entry) showed this endpoint's latency badly degrading under
    # concurrent load specifically because this made a fresh, uncached,
    # blocking network call on EVERY /health request regardless of
    # whether the outage was already known -- 20 concurrent health
    # checks against a down Ollama each individually paid the full 2s
    # timeout. core.circuit_breaker.ollama_judge_breaker already tracks
    # exactly this backend's health from the REAL judge-arbitration call
    # path; consulting it first means a known outage is reported
    # instantly, with zero network round-trip, instead of re-discovering
    # it the slow way on every single health check.
    #
    # Reads `_opened_at` directly rather than calling `is_open()` --
    # `is_open()` has a documented side effect (it transitions a
    # cooled-down breaker into its one-shot half-open probe state), and a
    # health check must never consume or reset that probe slot meant for
    # a real judge call. `core.metrics.refresh_circuit_breaker_gauges`
    # already established this exact pattern for the same reason.
    from core.circuit_breaker import ollama_judge_breaker
    with ollama_judge_breaker._lock:
        breaker_known_down = ollama_judge_breaker._opened_at is not None

    if breaker_known_down:
        status["status"] = "degraded"
    else:
        try:
            base_url = "/".join(OLLAMA_API_URL.split("/")[:-2])
            r = requests.get(f"{base_url}/tags", timeout=2)
            if r.status_code == 200:
                status["checks"]["semantic_judge"] = True
            else:
                status["status"] = "degraded"
        except Exception:
            status["status"] = "degraded"

    return status
