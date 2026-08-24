import json

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Literal, Optional

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
    decision: str = Field(
        ..., description="BLOCK, RESTRICT, ALLOW, or REVIEW. REVIEW means "
                         "neither auto-allowed nor auto-blocked — see "
                         "`review_id` and GET /api/v1/review/{review_id}."
    )
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
    review_id: Optional[str] = Field(
        None,
        description="Set when decision is REVIEW. Poll "
                    "GET /api/v1/review/{review_id} for the eventual "
                    "human decision — None for every other decision value.",
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


# --- Policy Editor (Phase 7) ---

class PolicyContentRequest(BaseModel):
    """Candidate policy file content, exactly as it would be written to
    disk (JSON or YAML text, matching the live file's own extension) --
    validated or deployed as a whole file, not a partial patch, mirroring
    how `scripts/manage_policy_versions.py deploy` already works.

    max_length=1_000_000 (1MB) matches this codebase's existing convention
    of bounding every free-text request field (see GatewayChatRequest's
    prompt/response_text below) -- a policy file is small, structured,
    repetitive JSON/YAML, so even a deployment with hundreds of tenants
    fits comfortably; this exists to bound the worst case (an oversized
    body used to waste memory/disk on the temp-file write this content
    goes through), not to constrain a realistic policy file."""
    content: str = Field(..., min_length=1, max_length=1_000_000)

    model_config = {"extra": "forbid"}


class PolicyRollbackRequest(BaseModel):
    # A version filename is `<timestamp>__<sha256[:12]>.<ext>` (see
    # core/policy_versioning.py) -- always well under 100 chars. Bounded
    # so an oversized value can't be used to probe filesystem behaviour
    # with an absurdly long path component.
    version: str = Field(..., min_length=1, max_length=255,
                        description="A version filename from GET /api/v1/policy's 'versions' list.")

    model_config = {"extra": "forbid"}


# --- Identity check (Phase 7) ---

class WhoAmIResponse(BaseModel):
    """What a caller's own credential resolves to. Deliberately the same
    shape as Principal.to_audit() -- capability/tenant/key_id only, never
    the credential itself. Exists so a client UI can validate a pasted API
    key against the real KeyStore (does it exist, is it still valid) rather
    than trusting whatever the browser happened to store."""
    capability: str
    tenant: str
    key_id: str

    model_config = {"extra": "forbid"}


# --- Human Review (Phase 4) ---

class ReviewStatusResponse(BaseModel):
    review_id: str
    status: str = Field(..., description="PENDING, APPROVED, or REJECTED.")
    reason: str = Field(..., description="Why this request was routed to review.")
    capability: str
    risk: str
    tenant: str
    request_id: str
    created_at: str
    resolved_at: Optional[str] = None
    reviewer: Optional[str] = Field(
        None, description="key_id of whoever resolved this — None while PENDING."
    )
    final_decision: Optional[str] = Field(
        None, description="ALLOW if APPROVED, BLOCK if REJECTED — None while PENDING."
    )


class ReviewResolveRequest(BaseModel):
    # Literal, not str -- an invalid value is rejected as a 422 by Pydantic
    # itself, before it ever reaches core.review_queue.resolve_review's
    # business logic. Keeps that function's own ValueError reserved for the
    # one case that IS a business-logic conflict: resolving an
    # already-resolved review.
    outcome: Literal["APPROVED", "REJECTED"] = Field(..., description="APPROVED or REJECTED.")

    model_config = {"extra": "forbid"}


# --- Real LLM Gateway (Phase 5) ---

class GatewayChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50_000,
                        description="The user prompt to send to the LLM, after Gatekeeper's own input guardrails.")
    system_prompt: Optional[str] = Field(
        None, max_length=20_000,
        description="Optional system prompt. Also used to check the response for "
                    "verbatim leakage of it, same as AssessRequest.system_prompt.",
    )
    provider: Optional[str] = Field(
        None, description="ollama, openai_compatible, or anthropic_compatible. "
                          "Defaults to LLM_GATEWAY_DEFAULT_PROVIDER if omitted.",
    )
    model: Optional[str] = Field(
        None, description="Model name for the chosen provider. Defaults to that "
                          "provider's own configured default if omitted.",
    )

    model_config = {"extra": "forbid"}


class GatewayChatResponse(BaseModel):
    decision: str = Field(..., description="The FINAL decision after both input and "
                                           "output guardrails: BLOCK, RESTRICT, ALLOW, or REVIEW.")
    content: Optional[str] = Field(
        None, description="The LLM's (PII-redacted) response, present only when "
                          "decision allowed it through. None for BLOCK/REVIEW, or "
                          "if the provider call itself failed.",
    )
    provider: Optional[str] = Field(None, description="Which provider actually handled this call.")
    model: Optional[str] = Field(None, description="Which model actually handled this call.")
    usage: Optional[Dict[str, Any]] = Field(
        None, description="Token usage as reported by the provider, verbatim -- "
                          "not yet enforced against any quota (Phase 5's "
                          "'Token accounting' item is unbuilt).",
    )
    review_id: Optional[str] = Field(None, description="Set when decision is REVIEW.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata and scores.")
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency, including the provider call.")


# --- Tool / Agent Gateway (Phase 6) ---

class ToolCallRequest(BaseModel):
    # max_length=200: registered tool names are short dotted identifiers
    # (e.g. "demo.database.query"); nothing in this codebase registers or
    # will plausibly register a longer one.
    name: str = Field(..., min_length=1, max_length=200,
                      description="The registered tool name to call, e.g. 'demo.database.query'.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool, validated against its declared JSON-Schema "
                    "parameters (core/tools.py::validate_arguments) before the tool runs.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("arguments")
    @classmethod
    def _bound_arguments_size(cls, value):
        """
        Phase 8 hardening: every other free-text field in this codebase
        already has a max_length (see AssessRequest.prompt, etc.) --
        `arguments` was the one caller-supplied, structurally-unbounded
        field in this schema file, since `core.tools.validate_arguments`
        only checks type/required/enum, never size (a `url` string
        argument for http.get, for instance, has no length cap at any
        layer). Bounded here at the schema boundary, before any of that
        downstream validation, hashing, or the tool's own handler ever
        sees it. 100_000 chars comfortably covers any real tool argument
        set (a URL, a table name, a row id) while still bounding the
        worst case (a deliberately oversized body wasting CPU/memory on
        JSON-dumping and hashing it before the tool even runs).
        """
        serialized_size = len(json.dumps(value, default=str))
        if serialized_size > 100_000:
            raise ValueError(
                f"arguments too large ({serialized_size} bytes serialized; max 100000)"
            )
        return value


class ToolCallResponse(BaseModel):
    decision: str = Field(..., description="BLOCK, REVIEW, or ALLOW -- core/tools.py's decision "
                                           "vocabulary, distinct from prompt-assessment's "
                                           "BLOCK/RESTRICT/ALLOW/REVIEW (RESTRICT has no meaning "
                                           "for a tool call that either runs or doesn't).")
    tool: str = Field(..., description="The tool name from the request, echoed back.")
    reason: str = Field(..., description="Why this decision was reached -- an access denial, a "
                                         "validation failure, 'HIGH risk; human approval required', "
                                         "or 'ok' for ALLOW.")
    output: Optional[Any] = Field(
        None, description="The handler's return value. Present only when decision is ALLOW "
                          "and the handler ran without raising.",
    )
    error: Optional[str] = Field(
        None, description="Set when decision is ALLOW but the handler itself raised -- "
                          "distinct from a BLOCK/REVIEW security decision. The call WAS "
                          "authorized to attempt running and failed on its own terms.",
    )
    review_id: Optional[str] = Field(
        None, description="Set when decision is REVIEW. Poll GET /api/v1/review/{review_id} "
                          "for the eventual human decision, same as a REVIEW from /api/v1/assess.",
    )
    process_time_ms: float = Field(default=0.0, description="Server-side processing latency in milliseconds.")

    model_config = {"extra": "forbid"}
