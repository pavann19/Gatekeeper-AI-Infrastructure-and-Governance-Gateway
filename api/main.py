from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
import time
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import AssessRequest, AssessResponse, AssessOutputRequest, AssessOutputResponse
from core.auth import auth_required, resolve_principal
from core.privacy import redact_pii
from core.rate_limit import assess_rate_limiter, bucket_parameters
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
    future = loop.run_in_executor(_assess_pool, functools.partial(func, *args))
    return await asyncio.wait_for(future, timeout=settings.ASSESS_TIMEOUT_SECONDS)


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


def _enforce_rate_limit(principal, request: Request) -> None:
    """
    Spends one token for this caller, or raises 429.

    Authenticated callers are keyed by their server-resolved `key_id`, which
    cannot be forged. Anonymous callers are keyed by peer address and given a
    smaller allowance, because their traffic is unattributable — there is no
    key to revoke if it turns out to be abusive.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    if principal.authenticated:
        identity = f"key:{principal.key_id}"
        rpm = settings.RATE_LIMIT_AUTHENTICATED_RPM
    else:
        identity = f"ip:{_client_address(request)}"
        rpm = settings.RATE_LIMIT_ANONYMOUS_RPM

    capacity, refill = bucket_parameters(rpm, settings.RATE_LIMIT_BURST_SECONDS)
    allowed, retry_after = assess_rate_limiter.check(identity, capacity, refill)

    if not allowed:
        # Log the identity, never the prompt: a rate-limit event is an
        # operational signal and must not become a second copy of user content.
        logger.warning(f"Rate limit exceeded for {identity} (limit {rpm}/min).")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit is {rpm:g} requests per minute.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    # Add timing header and make it accessible
    response.headers["X-Process-Time"] = str(process_time)
    return response

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

    # 0b. RATE LIMITING — before any expensive work is scheduled. Placing this
    #     after authentication is deliberate: the limit that applies depends on
    #     whether the caller has a verified identity.
    _enforce_rate_limit(principal, request)

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
            f"| key_id={principal.key_id}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Assessment did not complete within "
                f"{settings.ASSESS_TIMEOUT_SECONDS:g}s. The prompt was NOT "
                f"assessed and must not be treated as approved."
            ),
            headers={"Retry-After": "5"},
        )

    # 3. Policy Arbitration — using the SERVER-RESOLVED capability.
    decision, reason = policy_decision(principal.capability, risk_level)

    # Update details with reason and the identity the decision was made under.
    details["policy_reason"] = reason
    details["principal"] = principal.to_audit()

    # 4. Audit Logging
    log_event(principal.capability, clean_query, risk_level, decision, details)

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
        process_time_ms=process_time_ms
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
    _enforce_rate_limit(principal, request)

    logger.info(f"Output assessment request | key_id={principal.key_id}")

    from core.output_guardrails import assess_output

    # 1. Output Assessment (PII + Toxicity)
    try:
        decision, details = await _run_bounded(assess_output, req.response_text)
    except asyncio.TimeoutError:
        logger.error(
            f"Output assessment timed out after "
            f"{settings.ASSESS_TIMEOUT_SECONDS}s | key_id={principal.key_id}"
        )
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
