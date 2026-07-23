from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class AssessRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to analyze.")
    role: str = Field(default="GENERAL", description="The user capability tier.")

class AssessResponse(BaseModel):
    decision: str = Field(..., description="BLOCK, RESTRICT, or ALLOW")
    risk_level: str = Field(
        ...,
        description="Safety assessment: HIGH, MEDIUM, or LOW. Reflects threat "
                    "signals only — see `topicality` for subject-domain scoping.",
    )
    topicality: str = Field(
        default="UNKNOWN",
        description="Subject-domain scoping, independent of safety: IN_DOMAIN, "
                    "OUT_OF_DOMAIN, or UNKNOWN (guardrail disabled or not evaluated).",
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata and scores.")
    clean_prompt: str = Field(..., description="Prompt after PII redaction.")
    redacted_items: List[str] = Field(default_factory=list, description="List of redacted PII items.")
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency in milliseconds.")

class AssessOutputRequest(BaseModel):
    response_text: str = Field(..., description="The generated LLM response to analyze.")

class AssessOutputResponse(BaseModel):
    decision: str = Field(..., description="ALLOW or BLOCK")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata and scores.")
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency in milliseconds.")
