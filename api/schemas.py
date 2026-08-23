from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class AssessRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="The user prompt to analyze.",
    )

    # Optional. When present, /api/v1/assess runs BOTH input and output
    # assessment in one call — see api/main.py's "Output Guard" step for why
    # this exists: the Phase 0 V2 audit found Output Guard was a separate
    # endpoint a caller had to remember to invoke themselves, which is not a
    # property an MVP integration should require. This does NOT make
    # Gatekeeper call the caller's LLM — the caller still generates the
    # response and submits it here alongside the original prompt, in the
    # pattern "assess input -> call your own LLM -> submit both here". Bounded
    # identically to AssessOutputRequest.response_text for the same reason: a
    # limit either field could tolerate but the other couldn't would just
    # move the problem.
    response_text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50_000,
        description="Optional: the LLM's response to the prompt, if already "
                    "generated. When present, output assessment runs in the "
                    "same call and the response is assessed by the same "
                    "machinery as /api/v1/assess_output.",
    )

    # Optional reference text for system-prompt-leakage detection (Phase 2,
    # Output Security). Gatekeeper is a sidecar to the CALLER's own LLM call
    # and has no access to the caller's actual system prompt unless handed
    # one explicitly — there is nothing built into this gateway to detect
    # leakage against otherwise. Only meaningful together with response_text.
    system_prompt: Optional[str] = Field(
        None,
        max_length=20_000,
        description="Optional: the system prompt given to your LLM, if you "
                    "want the response checked for verbatim leakage of it. "
                    "Ignored if response_text is not also provided.",
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
    output_decision: Optional[str] = Field(
        None,
        description="ALLOW or BLOCK for the response_text, if one was "
                    "submitted. None when response_text was omitted — "
                    "distinct from ALLOW, which means it WAS checked and "
                    "passed.",
    )
    output_details: Optional[Dict[str, Any]] = Field(
        None, description="Output-assessment metadata, mirroring "
                          "AssessOutputResponse.details. None when "
                          "response_text was omitted.",
    )
    clean_response: Optional[str] = Field(
        None,
        description="response_text after PII redaction, if response_text "
                    "was submitted and contained no secrets/toxicity/"
                    "hallucination (those still BLOCK outright — there is "
                    "no safe partial version of a leaked credential). "
                    "None when response_text was omitted or the output was "
                    "blocked outright.",
    )

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

    # See AssessRequest.system_prompt — identical purpose, standalone-endpoint copy.
    system_prompt: Optional[str] = Field(
        None,
        max_length=20_000,
        description="Optional: the system prompt given to your LLM, if you "
                    "want the response checked for verbatim leakage of it.",
    )

    model_config = {"extra": "forbid"}

class AssessOutputResponse(BaseModel):
    decision: str = Field(..., description="ALLOW or BLOCK")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata and scores.")
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency in milliseconds.")
    clean_response: Optional[str] = Field(
        None,
        description="response_text after PII redaction. None when the "
                    "output was BLOCKed outright (secrets/toxicity/"
                    "hallucination have no safe partial version).",
    )
