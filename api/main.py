from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
import time
import asyncio
import functools
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from core import metrics
from api.schemas import AssessRequest, AssessResponse, AssessOutputRequest, AssessOutputResponse
from core.auth import auth_required, resolve_principal
from core.privacy import redact_pii
from core.rate_limit import assess_rate_limiter, bucket_parameters
from core.tenancy import resolve_tenant
from core.risk import assess_risk
from core.updates import fetch_latest_threats
from core.cache import flush_cache
from core.logger import get_logger, log_event
from core.policy import policy_decision
from core.config import settings

logger = get_logger("gatekeeper.api")

app = FastAPI(
    title="Gatekeeper AI Governance API",
    description="Neuro-Symbolic AI Security Gateway",
    version="2.0.0"
)

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
            assess_risk, clean_query, background_tasks.add_task
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
    #     BLOCKed — checking the output of a prompt that was never allowed
    #     through in the first place answers a question nobody asked.
    #
    #     The final decision is the MORE SEVERE of the two (BLOCK > RESTRICT
    #     > ALLOW): a clean prompt with a leaky or toxic response must still
    #     BLOCK — that is the entire point of checking output at all, not an
    #     edge case of it.
    output_decision = None
    output_details = None
    if req.response_text is not None and decision != "BLOCK":
        from core.output_guardrails import assess_output

        try:
            output_decision, output_details = await _run_bounded(
                assess_output, req.response_text
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

        _SEVERITY = {"ALLOW": 0, "RESTRICT": 1, "BLOCK": 2}
        if _SEVERITY.get(output_decision, 2) > _SEVERITY.get(decision, 0):
            decision = output_decision

    # Update details with reason and the identity the decision was made under.
    details["policy_reason"] = reason
    details["principal"] = principal.to_audit()
    details["request_id"] = request.state.request_id
    details["tenant"] = principal.tenant
    if output_details is not None:
        details["output_assessment"] = output_details

    # 4. Audit Logging — the FINAL combined decision, not the pre-output-check
    #    input decision, so a record never shows "ALLOW" for a request that
    #    was actually blocked on its response.
    log_event(principal.capability, clean_query, risk_level, decision, details)

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

    # 1. Output Assessment (PII + Toxicity)
    try:
        decision, details = await _run_bounded(assess_output, req.response_text)
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

    process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2) if hasattr(request.state, "start_time") else 0.0
    
    return AssessOutputResponse(
        decision=decision,
        details=details,
        process_time_ms=process_time_ms
    )

@app.post("/api/v1/update")
def update_threat_intel():
    count, success = fetch_latest_threats()
    if success:
        return {"status": "success", "signatures_added": count}
    raise HTTPException(status_code=500, detail="Failed to update threat intel.")

@app.post("/api/v1/cache/flush")
def flush_semantic_cache():
    flush_cache()
    return {"status": "success"}

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
    try:
        # Check base URL
        base_url = "/".join(OLLAMA_API_URL.split("/")[:-2])
        r = requests.get(f"{base_url}/tags", timeout=2)
        if r.status_code == 200:
            status["checks"]["semantic_judge"] = True
        else:
            status["status"] = "degraded"
    except Exception:
        status["status"] = "degraded"
        
    return status
