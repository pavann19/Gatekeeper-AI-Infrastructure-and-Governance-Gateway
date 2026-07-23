from fastapi import FastAPI, HTTPException, Request, Response
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import AssessRequest, AssessResponse, AssessOutputRequest, AssessOutputResponse
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
async def assess_prompt(req: AssessRequest, request: Request):
    request.state.start_time = time.perf_counter()
    logger.info(f"Received request for role: {req.role}")
    
    # 1. Privacy Layer
    clean_query, redacted_info = redact_pii(req.prompt)
    redacted_items = redacted_info.get("items", [])
    
    # 2. Risk Assessment (Neuro-Symbolic) - offloaded to thread to avoid blocking event loop
    risk_level, details = await asyncio.to_thread(assess_risk, clean_query)
    
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
        topicality=details.get("topicality", "UNKNOWN"),
        details=details,
        clean_prompt=clean_query,
        redacted_items=redacted_items,
        process_time_ms=process_time_ms
    )

@app.post("/api/v1/assess_output", response_model=AssessOutputResponse)
async def assess_output_endpoint(req: AssessOutputRequest, request: Request):
    request.state.start_time = time.perf_counter()
    logger.info("Received request for output assessment")
    
    from core.output_guardrails import assess_output
    
    # 1. Output Assessment (PII + Toxicity)
    decision, details = await asyncio.to_thread(assess_output, req.response_text)
    
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
