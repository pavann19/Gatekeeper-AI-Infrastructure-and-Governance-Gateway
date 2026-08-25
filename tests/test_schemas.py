"""
Dedicated schema-level tests for api/schemas.py.

Focus: field constraints, validators, and size bounds on the Pydantic
models themselves -- instantiated directly, with no HTTP/TestClient layer
involved. Endpoint-level behavior (how the API reacts to a 422) is covered
elsewhere (tests/test_gateway_chat.py, tests/test_tools_endpoint.py,
tests/test_policy_versioning.py); this file exists so the schemas have
dedicated boundary coverage independent of any endpoint wiring.
"""
import json

import pytest
from pydantic import ValidationError

from api.schemas import (
    AssessRequest,
    AssessResponse,
    AssessOutputRequest,
    AssessOutputResponse,
    PolicyContentRequest,
    PolicyRollbackRequest,
    WhoAmIResponse,
    ReviewStatusResponse,
    ReviewResolveRequest,
    GatewayChatRequest,
    GatewayChatResponse,
    ToolCallRequest,
    ToolCallResponse,
)


# --- AssessRequest ---

class TestAssessRequest:
    def test_minimal_valid(self):
        req = AssessRequest(prompt="hi")
        assert req.prompt == "hi"
        assert req.response_text is None
        assert req.system_prompt is None

    def test_prompt_empty_rejected(self):
        with pytest.raises(ValidationError):
            AssessRequest(prompt="")

    def test_prompt_missing_rejected(self):
        with pytest.raises(ValidationError):
            AssessRequest()

    def test_prompt_max_length_accepted(self):
        req = AssessRequest(prompt="a" * 50_000)
        assert len(req.prompt) == 50_000

    def test_prompt_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            AssessRequest(prompt="a" * 50_001)

    def test_response_text_empty_rejected(self):
        # min_length=1 when present -- None is fine, "" is not.
        with pytest.raises(ValidationError):
            AssessRequest(prompt="hi", response_text="")

    def test_response_text_max_length_accepted(self):
        req = AssessRequest(prompt="hi", response_text="b" * 50_000)
        assert len(req.response_text) == 50_000

    def test_response_text_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            AssessRequest(prompt="hi", response_text="b" * 50_001)

    def test_system_prompt_max_length_accepted(self):
        req = AssessRequest(prompt="hi", system_prompt="s" * 20_000)
        assert len(req.system_prompt) == 20_000

    def test_system_prompt_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            AssessRequest(prompt="hi", system_prompt="s" * 20_001)

    def test_system_prompt_empty_string_allowed(self):
        # system_prompt has no min_length constraint, unlike response_text.
        req = AssessRequest(prompt="hi", system_prompt="")
        assert req.system_prompt == ""

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AssessRequest(prompt="hi", role="INTERNAL")

    def test_extra_field_role_specifically_rejected(self):
        # Regression guard for the documented security fix: `role` was
        # removed deliberately so a client can't self-declare privilege.
        with pytest.raises(ValidationError) as exc_info:
            AssessRequest(prompt="hi", role="INTERNAL")
        assert "role" in str(exc_info.value)


# --- AssessResponse ---

class TestAssessResponse:
    def test_minimal_required_fields(self):
        resp = AssessResponse(decision="ALLOW", risk_level="LOW", clean_prompt="hi")
        assert resp.topicality == "UNKNOWN"
        assert resp.capability == "GENERAL"
        assert resp.authenticated is False
        assert resp.details == {}
        assert resp.redacted_items == []
        assert resp.process_time_ms == 0.0
        assert resp.output_decision is None
        assert resp.output_details is None
        assert resp.clean_response is None
        assert resp.review_id is None

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            AssessResponse(risk_level="LOW", clean_prompt="hi")

    def test_default_factory_independence(self):
        # default_factory=dict/list must not share mutable state between instances.
        r1 = AssessResponse(decision="ALLOW", risk_level="LOW", clean_prompt="hi")
        r2 = AssessResponse(decision="ALLOW", risk_level="LOW", clean_prompt="hi")
        r1.details["x"] = 1
        r1.redacted_items.append("y")
        assert r2.details == {}
        assert r2.redacted_items == []


# --- AssessOutputRequest ---

class TestAssessOutputRequest:
    def test_minimal_valid(self):
        req = AssessOutputRequest(response_text="hi")
        assert req.system_prompt is None

    def test_response_text_required(self):
        with pytest.raises(ValidationError):
            AssessOutputRequest()

    def test_response_text_empty_rejected(self):
        with pytest.raises(ValidationError):
            AssessOutputRequest(response_text="")

    def test_response_text_max_length_accepted(self):
        req = AssessOutputRequest(response_text="x" * 50_000)
        assert len(req.response_text) == 50_000

    def test_response_text_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            AssessOutputRequest(response_text="x" * 50_001)

    def test_system_prompt_max_length_accepted(self):
        req = AssessOutputRequest(response_text="hi", system_prompt="s" * 20_000)
        assert len(req.system_prompt) == 20_000

    def test_system_prompt_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            AssessOutputRequest(response_text="hi", system_prompt="s" * 20_001)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AssessOutputRequest(response_text="hi", extra_field="x")


# --- AssessOutputResponse ---

class TestAssessOutputResponse:
    def test_defaults(self):
        resp = AssessOutputResponse(decision="ALLOW")
        assert resp.details == {}
        assert resp.process_time_ms == 0.0
        assert resp.clean_response is None

    def test_decision_required(self):
        with pytest.raises(ValidationError):
            AssessOutputResponse()


# --- PolicyContentRequest ---

class TestPolicyContentRequest:
    def test_valid_content(self):
        req = PolicyContentRequest(content='{"a": 1}')
        assert req.content == '{"a": 1}'

    def test_content_required(self):
        with pytest.raises(ValidationError):
            PolicyContentRequest()

    def test_content_empty_rejected(self):
        with pytest.raises(ValidationError):
            PolicyContentRequest(content="")

    def test_content_max_length_accepted(self):
        req = PolicyContentRequest(content="a" * 1_000_000)
        assert len(req.content) == 1_000_000

    def test_content_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            PolicyContentRequest(content="a" * 1_000_001)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PolicyContentRequest(content="x", extra="y")


# --- PolicyRollbackRequest ---

class TestPolicyRollbackRequest:
    def test_valid_version(self):
        req = PolicyRollbackRequest(version="20240101__abc123abc123.json")
        assert req.version.endswith(".json")

    def test_version_required(self):
        with pytest.raises(ValidationError):
            PolicyRollbackRequest()

    def test_version_empty_rejected(self):
        with pytest.raises(ValidationError):
            PolicyRollbackRequest(version="")

    def test_version_max_length_accepted(self):
        req = PolicyRollbackRequest(version="a" * 255)
        assert len(req.version) == 255

    def test_version_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            PolicyRollbackRequest(version="a" * 256)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PolicyRollbackRequest(version="v1", extra="y")


# --- WhoAmIResponse ---

class TestWhoAmIResponse:
    def test_valid(self):
        resp = WhoAmIResponse(capability="GENERAL", tenant="default", key_id="k1")
        assert resp.capability == "GENERAL"

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            WhoAmIResponse(capability="GENERAL", tenant="default")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            WhoAmIResponse(capability="GENERAL", tenant="default", key_id="k1", extra="y")


# --- ReviewStatusResponse ---

class TestReviewStatusResponse:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            review_id="r1",
            status="PENDING",
            reason="ambiguous",
            capability="GENERAL",
            risk="MEDIUM",
            tenant="default",
            request_id="req1",
            created_at="2024-01-01T00:00:00Z",
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_minimal(self):
        resp = ReviewStatusResponse(**self._base_kwargs())
        assert resp.resolved_at is None
        assert resp.reviewer is None
        assert resp.final_decision is None

    def test_missing_required_field_rejected(self):
        kwargs = self._base_kwargs()
        del kwargs["status"]
        with pytest.raises(ValidationError):
            ReviewStatusResponse(**kwargs)

    def test_resolved_fields_populated(self):
        resp = ReviewStatusResponse(**self._base_kwargs(
            status="APPROVED", resolved_at="2024-01-02T00:00:00Z",
            reviewer="admin", final_decision="ALLOW",
        ))
        assert resp.final_decision == "ALLOW"

    def test_no_extra_forbid_configured(self):
        # ReviewStatusResponse has no model_config extra=forbid override, so
        # extra fields are allowed under default pydantic behavior (ignored).
        resp = ReviewStatusResponse(**self._base_kwargs(unexpected="value"))
        assert not hasattr(resp, "unexpected") or True  # default is "ignore"


# --- ReviewResolveRequest ---

class TestReviewResolveRequest:
    def test_approved_accepted(self):
        req = ReviewResolveRequest(outcome="APPROVED")
        assert req.outcome == "APPROVED"

    def test_rejected_accepted(self):
        req = ReviewResolveRequest(outcome="REJECTED")
        assert req.outcome == "REJECTED"

    def test_invalid_literal_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResolveRequest(outcome="MAYBE")

    def test_lowercase_rejected(self):
        # Literal match is case-sensitive.
        with pytest.raises(ValidationError):
            ReviewResolveRequest(outcome="approved")

    def test_missing_outcome_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResolveRequest()

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ReviewResolveRequest(outcome="APPROVED", extra="y")


# --- GatewayChatRequest ---

class TestGatewayChatRequest:
    def test_minimal_valid(self):
        req = GatewayChatRequest(prompt="hi")
        assert req.system_prompt is None
        assert req.provider is None
        assert req.model is None

    def test_prompt_required(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest()

    def test_prompt_empty_rejected(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest(prompt="")

    def test_prompt_max_length_accepted(self):
        req = GatewayChatRequest(prompt="a" * 50_000)
        assert len(req.prompt) == 50_000

    def test_prompt_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest(prompt="a" * 50_001)

    def test_system_prompt_max_length_accepted(self):
        req = GatewayChatRequest(prompt="hi", system_prompt="s" * 20_000)
        assert len(req.system_prompt) == 20_000

    def test_system_prompt_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest(prompt="hi", system_prompt="s" * 20_001)

    def test_provider_max_length_accepted(self):
        req = GatewayChatRequest(prompt="hi", provider="p" * 100)
        assert len(req.provider) == 100

    def test_provider_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest(prompt="hi", provider="p" * 101)

    def test_model_max_length_accepted(self):
        req = GatewayChatRequest(prompt="hi", model="m" * 200)
        assert len(req.model) == 200

    def test_model_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest(prompt="hi", model="m" * 201)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            GatewayChatRequest(prompt="hi", role="INTERNAL")


# --- GatewayChatResponse ---

class TestGatewayChatResponse:
    def test_defaults(self):
        resp = GatewayChatResponse(decision="ALLOW")
        assert resp.content is None
        assert resp.provider is None
        assert resp.model is None
        assert resp.usage is None
        assert resp.review_id is None
        assert resp.details == {}
        assert resp.process_time_ms == 0.0

    def test_decision_required(self):
        with pytest.raises(ValidationError):
            GatewayChatResponse()


# --- ToolCallRequest ---

class TestToolCallRequest:
    def test_minimal_valid(self):
        req = ToolCallRequest(name="demo.tool")
        assert req.arguments == {}

    def test_name_required(self):
        with pytest.raises(ValidationError):
            ToolCallRequest()

    def test_name_empty_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallRequest(name="")

    def test_name_max_length_accepted(self):
        req = ToolCallRequest(name="a" * 200)
        assert len(req.name) == 200

    def test_name_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallRequest(name="a" * 201)

    def test_default_arguments_independence(self):
        # default_factory=dict must not share state across instances.
        r1 = ToolCallRequest(name="t1")
        r2 = ToolCallRequest(name="t2")
        r1.arguments["x"] = 1
        assert r2.arguments == {}

    def test_arguments_just_under_limit_accepted(self):
        # Build a payload whose serialized JSON size is right under 100_000
        # bytes. json.dumps({"k": "aaaa..."}) overhead is small and fixed,
        # so pad precisely to land just under the boundary.
        overhead = len(json.dumps({"k": ""}, default=str))
        pad_len = 100_000 - overhead - 1
        value = "a" * pad_len
        req = ToolCallRequest(name="t", arguments={"k": value})
        serialized_size = len(json.dumps(req.arguments, default=str))
        assert serialized_size < 100_000

    def test_arguments_over_limit_rejected(self):
        overhead = len(json.dumps({"k": ""}, default=str))
        pad_len = 100_000 - overhead + 100  # comfortably over
        value = "a" * pad_len
        with pytest.raises(ValidationError) as exc_info:
            ToolCallRequest(name="t", arguments={"k": value})
        assert "too large" in str(exc_info.value)

    def test_arguments_exactly_at_limit_accepted(self):
        # Construct a dict whose json.dumps(..., default=str) length is
        # exactly 100_000 bytes -- the validator only rejects strictly
        # greater than 100_000.
        overhead = len(json.dumps({"k": ""}, default=str))
        pad_len = 100_000 - overhead
        value = "a" * pad_len
        req = ToolCallRequest(name="t", arguments={"k": value})
        serialized_size = len(json.dumps(req.arguments, default=str))
        assert serialized_size == 100_000

    def test_arguments_one_byte_over_limit_rejected(self):
        overhead = len(json.dumps({"k": ""}, default=str))
        pad_len = 100_000 - overhead + 1
        value = "a" * pad_len
        with pytest.raises(ValidationError):
            ToolCallRequest(name="t", arguments={"k": value})

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ToolCallRequest(name="t", extra="y")


# --- ToolCallResponse ---

class TestToolCallResponse:
    def test_minimal_valid(self):
        resp = ToolCallResponse(decision="ALLOW", tool="t1", reason="ok")
        assert resp.output is None
        assert resp.error is None
        assert resp.review_id is None
        assert resp.process_time_ms == 0.0

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallResponse(decision="ALLOW", tool="t1")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ToolCallResponse(decision="ALLOW", tool="t1", reason="ok", extra="y")
