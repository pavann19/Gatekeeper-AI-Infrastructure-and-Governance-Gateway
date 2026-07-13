from fastapi import FastAPI, HTTPException, Request, Response
import time
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import AssessRequest, AssessResponse
from core.privacy import redact_pii
from core.risk import assess_risk
from core.updates import fetch_latest_threats
from core.cache import flush_cache
from core.logger import get_logger, log_event
from core.policy import policy_decision

logger = get_logger("gatekeeper.api")

app = FastAPI(
    title="Gatekeeper AI Governance API",
    description="Neuro-Symbolic AI Security Gateway",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
def assess_prompt(req: AssessRequest, request: Request):
    request.state.start_time = time.perf_counter()
    logger.info(f"Received request for role: {req.role}")
    
    # 1. Privacy Layer
    clean_query, redacted_info = redact_pii(req.prompt)
    redacted_items = redacted_info.get("items", [])
    
    # 2. Risk Assessment (Neuro-Symbolic)
    risk_level, details = assess_risk(clean_query)
    
    # 3. Policy Arbitration
    decision, reason = policy_decision(req.role, risk_level)
    
    # Update details with reason
    details["policy_reason"] = reason
    
    # 4. Audit Logging
    log_event(req.role, clean_query, risk_level, decision, details)
    
    # Calculate process time explicitly for the response body (if needed by client directly)
    # The header is already set by middleware, but this makes it visible in the JSON
    # We'll just pass 0.0 here since the middleware calculates the true end-to-end,
    # but the client can read the header. Alternatively, we can calculate it here:
    process_time_ms = round((time.perf_counter() - request.state.start_time) * 1000, 2) if hasattr(request.state, "start_time") else 0.0
    
    return AssessResponse(
        decision=decision,
        risk_level=risk_level,
        details=details,
        clean_prompt=clean_query,
        redacted_items=redacted_items,
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
    return {"status": "healthy"}
