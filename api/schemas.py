from pydantic import BaseModel, Field
from typing import Dict, Any, List

class AssessRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="The user prompt to analyze.",
    )

    # NOTE: `role` was REMOVED from this schema deliberately, and its absence is
    # a security control rather than an oversight.
    #
    # It previously carried the caller's capability tier, and api/main.py passed
    # it straight to the policy engine. Since INTERNAL maps HIGH -> ALLOW, any
    # client could bypass every guardrail by sending {"role": "INTERNAL"}.
    #
    # Capability is now resolved server-side from a verified API key. A client
    # may present a credential; it may not declare its own privilege. Extra
    # fields are rejected so that a caller still sending `role` gets a loud 422
    # rather than silently believing it had an effect.
    model_config = {"extra": "forbid"}

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
    capability: str = Field(
        default="GENERAL",
        description="Capability tier the policy was evaluated under, resolved "
                    "server-side from the presented credential. Echoed so the "
                    "caller can see what privilege it actually had.",
    )
    authenticated: bool = Field(
        default=False,
        description="Whether a valid API key was presented. False means the "
                    "request was served anonymously at least privilege.",
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata and scores.")
    clean_prompt: str = Field(..., description="Prompt after PII redaction.")
    redacted_items: List[str] = Field(default_factory=list, description="List of redacted PII items.")
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency in milliseconds.")

class AssessOutputRequest(BaseModel):
    # Bounded for the same reason AssessRequest.prompt is: this text is
    # embedded and run through the output guardrails, so an unbounded field
    # lets one caller pin a worker from the (deliberately small) assessment
    # pool for as long as it takes to process a multi-megabyte body. The cap
    # matches AssessRequest deliberately — a response is assessed by the same
    # machinery as a prompt, so a limit either side could tolerate but the
    # other could not would just move the problem.
    response_text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="The generated LLM response to analyze.",
    )

    model_config = {"extra": "forbid"}

class AssessOutputResponse(BaseModel):
    decision: str = Field(..., description="ALLOW or BLOCK")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata and scores.")
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency in milliseconds.")
