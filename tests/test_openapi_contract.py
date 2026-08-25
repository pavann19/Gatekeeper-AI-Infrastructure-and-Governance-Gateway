"""
Contract conformance: the OpenAPI schema FastAPI generates from api/schemas.py
must actually document the security-relevant size bounds this project's
Phase 8 hardening work enforces (core/policy_versioning.py's path-traversal
fix's sibling size-bound work, the tool/gateway maxLength additions). This
is a real regression guard: if a future edit removes a `Field(max_length=...)`
constraint, the documented contract silently weakens right along with the
enforcement, and this is the test that would catch it.

Also validates the schema is spec-conformant OpenAPI 3.1, not just
"FastAPI didn't crash generating it" -- a schema can be internally
inconsistent (bad $ref, wrong type keyword) without FastAPI itself erroring.
"""
from openapi_spec_validator import validate

from api.main import app


def _get_openapi_schema():
    # FastAPI caches app.openapi_schema after the first call; clear it so
    # repeated test runs (and any prior real request in this process) don't
    # return a stale cached copy from before a code change.
    app.openapi_schema = None
    return app.openapi()


def _get_constraint(prop: dict, key: str):
    """Pydantic v2 represents an Optional[str] field's constraints nested
    inside an OpenAPI 3.1 `anyOf: [{type, maxLength}, {type: null}]`, not as
    a flat property -- this looks in both places."""
    if key in prop:
        return prop[key]
    for sub in prop.get("anyOf", []):
        if key in sub:
            return sub[key]
    return None


def test_generated_schema_is_valid_openapi_3_1():
    schema = _get_openapi_schema()
    assert schema.get("openapi", "").startswith("3.1")
    validate(schema)  # raises on any spec violation


def test_every_endpoint_is_documented():
    schema = _get_openapi_schema()
    paths = schema.get("paths", {})
    # A real regression guard against accidentally excluding a live route
    # from the schema (e.g. a stray include_in_schema=False) -- the
    # metrics endpoint is the one deliberate exception (see its own
    # docstring for why).
    assert "/api/v1/assess" in paths
    assert "/api/v1/policy/rollback" in paths
    assert "/api/v1/tools/call" in paths
    assert "/health" in paths


def test_documented_size_bounds_match_enforced_validation():
    schema = _get_openapi_schema()
    schemas = schema["components"]["schemas"]

    checks = [
        ("PolicyContentRequest", "content", "maxLength", 1_000_000),
        ("PolicyRollbackRequest", "version", "maxLength", 255),
        ("ToolCallRequest", "name", "maxLength", 200),
        ("GatewayChatRequest", "model", "maxLength", 200),
        ("GatewayChatRequest", "provider", "maxLength", 100),
    ]
    for model, field, constraint, expected in checks:
        prop = schemas[model]["properties"][field]
        actual = _get_constraint(prop, constraint)
        assert actual == expected, (
            f"{model}.{field}.{constraint} documented as {actual}, "
            f"expected {expected} -- the OpenAPI contract has drifted from "
            f"the real enforced validation"
        )


def test_extra_forbid_schemas_reject_unknown_fields_in_the_contract():
    """Every request schema that sets extra="forbid" (rejecting unknown
    fields at the Pydantic layer) must document additionalProperties:
    false in the OpenAPI contract too -- a caller reading only the docs
    should see the same "no extra fields" rule the API actually enforces."""
    schema = _get_openapi_schema()
    schemas = schema["components"]["schemas"]

    forbid_schemas = ["AssessRequest", "GatewayChatRequest", "ToolCallRequest",
                       "PolicyContentRequest", "PolicyRollbackRequest"]
    for name in forbid_schemas:
        assert schemas[name].get("additionalProperties") is False, (
            f"{name} should document additionalProperties: false to match "
            f"its real extra='forbid' enforcement"
        )
