# Integration guide

Use the [deployment guide](DEPLOYMENT_RUNBOOK.md) to provision the service and credentials. Open `/docs` for schemas and `/openapi.json` for the current generated contract. The examples below are illustrative requests, not recorded model outcomes or latency measurements.

## HTTP identity

Send `Authorization: Bearer <key>` or `X-API-Key`. Capability and tenant come from the credential. Do not send a role/capability field to grant yourself permission. Required authentication returns 401 for anonymous callers; optional mode permits GENERAL access where allowed. Tenant suspension/unavailable admission denies requests. INTERNAL is a trusted administrative credential, not a standard application key.

For POSIX request examples, place the issued key in `GATEKEEPER_API_KEY`. On PowerShell use `$env:GATEKEEPER_API_KEY` and `Invoke-RestMethod` or `curl.exe` with the equivalent headers; shell quoting differs.

## Assessment and decisions

```sh
curl -sS http://localhost:8000/api/v1/assess \
  -H "Authorization: Bearer $GATEKEEPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"How do I reset my password?"}'
```

The response includes `decision`, `risk_level`, `topicality`, `clean_prompt`, redaction metadata, details and request identity. Risk describes threat assessment; topicality describes subject-domain scope. Model-dependent verdicts are configuration-dependent.

| Decision | Caller behavior |
|---|---|
| ALLOW | Continue only with the assessed/privacy-processed content and required downstream checks |
| BLOCK | Do not execute or forward |
| RESTRICT | Do not treat as unrestricted permission; enforce the applicable restriction or withhold execution |
| REVIEW | Hold execution; resolving a record is not general permission for a later action |

For an application calling its own model, assess input first, enforce the decision, then assess generated output before releasing it. `/api/v1/assess` accepts optional `response_text` and reference `system_prompt` for combined assessment of an already generated exchange. `/api/v1/assess_output` checks output independently:

```sh
curl -sS http://localhost:8000/api/v1/assess_output \
  -H "Authorization: Bearer $GATEKEEPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"response_text":"Use the account recovery page."}'
```

Do not confuse absent output assessment (`output_decision` unset) with output acceptance. Secret/leak checks, PII processing and explicit SAFE toxicity judging apply. Unavailable required judging blocks output. The optional grounding check is not a general factuality guarantee.

## Provider gateway

```sh
curl -sS http://localhost:8000/api/v1/gateway/chat \
  -H "Authorization: Bearer $GATEKEEPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain password recovery.","provider":"ollama"}'
```

Configure provider endpoints, credentials and models on the server. Input BLOCK/RESTRICT/REVIEW stops forwarding; output must pass before content is released. Provider usage supports token accounting, subject to provider reporting and concurrent quota limitations. Errors are not successful model responses. Gateway system prompts bypass the user-prompt screening path: source them from trusted application configuration rather than untrusted user input.

## Tools and review

Operators register tools with capability, risk, schema and a trusted handler. Demo/real tools are opt-in. Inspect `/api/v1/tools` and the generated schema for the current catalogue. An enabled demo example is:

```sh
curl -sS http://localhost:8000/api/v1/tools/call \
  -H "Authorization: Bearer $GATEKEEPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"demo.echo","arguments":{"text":"hello"}}'
```

HIGH tools return REVIEW before effects. Authenticated tenant owners can inspect records; ELEVATED/INTERNAL reviewers can resolve owned records. Use `GET /api/v1/review/{review_id}` and `POST /api/v1/review/{review_id}/resolve` with `{"outcome":"APPROVED"}` or `{"outcome":"REJECTED"}`. INTERNAL has deliberate cross-tenant administration.

For approved tool continuation, resubmit the identical tool/schema/arguments using the same authenticated principal and original `X-Request-ID`, adding `review_id` to the tool request. Tenant, capability, policy/security identity and decision context must still match. Approval is consumed once before effects; changed requests, expired/conflicting records and previously consumed approvals do not authorize execution. Investigate uncertainty after a crash instead of retrying an effect blindly.

Prompt/provider reviews remain decision records; they have no continuation endpoint. Do not turn an approved record into a generic client-side ALLOW override. Registered handlers are trusted code, not a sandbox.

## MCP

MCP transports expose tool discovery/execution, not general prompt assessment:

```sh
python -m scripts.run_mcp_server --capability GENERAL --tenant default --demo-tools
python -m scripts.run_mcp_http_server --host 127.0.0.1 --port 8001 --demo-tools
```

Stdio trusts its launcher and fixed principal. HTTP requires configured credentials and tenant admission even when the main API permits optional authentication. `/mcp/jsonrpc` returns stateless JSON-RPC responses. `/mcp/sse` establishes a credential-bound session; `/mcp/messages` relays through that session. Keep the same credentials and principal for session use. A session identifier alone does not grant access. Shared tool controls still apply; do not assume HTTP tool approval continuation is exposed as an equivalent MCP protocol feature.

## Failure and operational handling

Use `/live` for process liveness and `/ready` for readiness. `/health` is diagnostic. Treat 401/403, 429, body/schema errors, dependency failures and timeouts explicitly. A timed-out tool/provider operation is not proof that external execution never occurred. Preserve `X-Request-ID` for correlation and do not put credentials or raw sensitive content in support reports.

Request deadlines and backend concurrency tests do not certify every synchronous/nested path or application replica configuration. Read [security](SECURITY.md) before enabling remote providers, real network tools or public access.
