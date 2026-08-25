# Gatekeeper — Threat Model

Phase 8 (Production Hardening) deliverable. This document does not
describe an aspirational design — every control listed here is real code
in this repository, cited by file and function, and every finding is
either something this project's own security-audit passes (Phase 8,
`docs/ROADMAP_V2.md`) actually found and fixed, or a limitation that
codebase already states explicitly in its own docstrings. Nothing here
is invented for the document; it consolidates what the code already does
and admits, in one place, structured the way a reviewer would want to
read it.

Scope: the HTTP API (`api/main.py`) and the subsystems it exposes —
detection pipeline, tool/agent gateway, LLM gateway, policy engine,
human review queue, and the client UI. The MCP stdio transport
(`core/mcp_server.py`) and MCP HTTP/SSE server (`core/mcp_http_server.py`)
are covered in §6.

---

## 1. System overview and trust boundaries

```
                    ┌─────────────────────────────────────────┐
                    │              Caller (untrusted)           │
                    └──────────────────┬────────────────────────┘
                                       │ HTTPS (operator-terminated)
                    ┌──────────────────▼────────────────────────┐
                    │   api/main.py — the ONLY trust boundary    │
                    │   crossing: everything past this point     │
                    │   trusts what it resolved, not what the    │
                    │   caller asserted (core/auth.py)            │
                    └──────────────────┬────────────────────────┘
         ┌─────────────┬───────────────┼───────────────┬─────────────┐
         ▼             ▼               ▼               ▼             ▼
   Detection      Tool/Agent      LLM Gateway      Policy Engine   Human
   pipeline       Gateway         (proxied call    (capability +   Review
   (core/risk.py) (core/tools.py) to a real         tenant ->       Queue
                  + core/         provider)         action)        (core/
                  real_tools.py)                                   review_
                                                                    queue.py)
```

**The one real trust boundary is `core/auth.py::resolve_principal`.**
Every other module in this diagram trusts the `capability`/`tenant` it
receives *because* it came from that function, never because a caller
asserted it. `api/schemas.py::AssessRequest`'s own docstring records why
this matters: an earlier version of this schema accepted a `role` field
straight from the caller and passed it to the policy engine, which meant
`{"role": "INTERNAL"}` bypassed every guardrail. That field was removed
and `extra="forbid"` added specifically so a client still sending `role`
gets a loud 422 instead of silently believing it had an effect. This
threat model assumes that discipline holds everywhere else in the
codebase too, and §7 is exactly the record of checking that assumption
project-wide rather than trusting it by default.

## 2. Actors and capability tiers

Three tiers, ranked (`core/tools.py::CAPABILITY_RANK`):

| Capability | Rank | Typical actor | What it can reach |
|---|---|---|---|
| `GENERAL` | 0 | Anonymous caller, or a low-trust API key | Assess/gateway endpoints at the strictest policy tier; own-tenant activity/trace/settings |
| `ELEVATED` | 1 | A trusted integration | Same as GENERAL plus MEDIUM-risk tools (`http.get`), looser policy mapping |
| `INTERNAL` | 2 | An operator / the system itself | Everything ELEVATED can, plus cross-tenant activity/logs, the review queue, the policy editor, the tool/gateway catalogues |

A capability is never asserted by the caller — it is resolved
server-side from a verified API key (`core/auth.py::KeyStore`, keys
stored as SHA-256 hashes, never in plaintext) or defaults to `GENERAL`
for an anonymous caller (fail-open to the LEAST privileged tier, not the
most). `AUTH_MODE=required` rejects anonymous callers outright;
`AUTH_MODE=optional` (default) serves them at `GENERAL`.

A fourth "actor" this model treats distinctly: **a tenant's own
suspended state.** `core.tenancy.TenantConfig.suspended` is orthogonal to
capability — an INTERNAL key belonging to a suspended tenant is still
rejected (`api/main.py::_reject_suspended_and_rate_limit`, §7 finding
#2). Suspension is an operator action, not a capability downgrade.

## 3. Assets

| Asset | Where it lives | Protected by |
|---|---|---|
| API keys | `core/auth.py::KeyStore` (hashed) | SHA-256 at rest, never logged, never returned after issuance (`scripts/manage_api_keys.py`'s own "shown once" banner) |
| Prompt/response content | Never persisted raw | Audit log stores only `sha256(prompt)`/`sha256(response)` (`core/logger.py::log_event`/`log_output_event`) |
| Tool call arguments | Never persisted raw | Same hash-only discipline (`core/logger.py::log_tool_event`), extended in Phase 6 |
| The live policy | `policy_rules.json` (or `.yaml`) | `core/policy_versioning.py`'s snapshot-before-overwrite discipline; every deploy validated first (§4) |
| Tenant SLA/status | `tenants.json` | Operator-provisioned only, never caller-writable |
| PII in transit | Regex + NER pipeline (`core/privacy.py`) | Redacted before logging or forwarding, per-tenant overrides supported |

## 4. STRIDE-style analysis by subsystem

### 4.1 Detection pipeline (`/api/v1/assess`, `/api/v1/assess_output`)

- **Tampering / Denial of Service** — every free-text field is
  length-bounded (`AssessRequest.prompt` 50k chars,
  `system_prompt`/`response_text` 20k) and the assessment worker pool is
  bounded (`ASSESS_MAX_CONCURRENCY`) with a hard deadline
  (`ASSESS_TIMEOUT_SECONDS`) that fails closed to 503, never a fabricated
  verdict. A judge backend circuit breaker (`core/circuit_breaker.py`)
  stops a failing Ollama/Llama Guard instance from making every
  ambiguous-zone request pay its full timeout individually.
- **Information Disclosure** — PII is redacted before it reaches the
  audit log or any response (`core/privacy.py`); secrets in LLM output
  are detected and blocked, never partially redacted-and-returned
  (`core/secrets_detection.py`).
- **Repudiation** — every decision is audited with a `request_id`
  correlating it back to the HTTP request (`core/logger.py`,
  `_resolve_request_id`), joinable across the input assessment, output
  assessment, gateway call, and tool call event types.
- **Tampering (of the judge itself)**, §7 finding #10: the judge
  arbitration step (`core/semantic_judge.py`) is itself an LLM call, and
  its non-Llama-Guard fallback path concatenated the untrusted
  prompt/response directly into the judge's own instruction text with no
  protocol-level separation — a real prompt-injection surface against
  the classifier deciding SAFE/DANGEROUS. Mitigated with delimiter tags
  and explicit "treat as data, not instructions" framing.

### 4.2 Tool/Agent Gateway (`/api/v1/tools/call`, `core/tools.py`)

- **Elevation of Privilege** — a tool call is decided in a fixed order
  (access control → structural validation → risk-based approval),
  cheapest and most decisive first (`core/tools.py::decide_tool_call`'s
  own docstring). A HIGH-risk tool always requires human REVIEW even for
  a caller whose capability already clears access control.
- **Tampering** — `core/real_tools.py`'s `http.get` (the one tool that
  makes a real outbound network call) is the SSRF case study: a
  hostname allowlist alone is insufficient against DNS rebinding, so
  every call also resolves the hostname and rejects it if ANY address is
  private/loopback/link-local/reserved, checked at call time, not once
  at startup. Redirects are never followed. Response size and request
  duration are both capped.
- **Denial of Service** — §7 finding #3: `ToolCallRequest.arguments` has
  a 100KB serialized cap plus per-field JSON-Schema `maxLength` support in
  `validate_arguments`, applied concretely to `http.get`'s `url` (2048 chars).

### 4.3 LLM Gateway (`/api/v1/gateway/chat`, `core/llm_providers.py`)

- **Spoofing** — the outbound provider endpoint (`base_url`) is always
  server-configured (`settings.OLLAMA_CHAT_URL` / `OPENAI_BASE_URL` /
  `ANTHROPIC_BASE_URL`); `get_provider(name)` takes no caller-supplied
  URL override.
- **Denial of Service** — `model` and `provider` length bounded (200 / 100 chars).
- **Information Disclosure** — token usage is metered per-tenant
  (`core/token_quota.py`) only from what a provider actually reports.

### 4.4 Policy Engine and Policy Editor (`/api/v1/policy/*`)

- **Tampering** — §7 finding #1: `core.policy_versioning.rollback_to`
  rejects any `version_name` that contains path traversal characters or
  isn't a bare filename.
- **Elevation of Privilege** — deploy is validated BEFORE it is ever
  written (`validate_policy_file`), and every deploy/rollback snapshots
  the outgoing policy first.
- **Denial of Service** — `PolicyContentRequest.content` is capped at 1MB,
  `PolicyRollbackRequest.version` at 255 chars.

### 4.5 Human Review Queue (`/api/v1/review/*`)

- **Information Disclosure** — a review record stores only a
  `prompt_hash`/argument hash, never raw content (`core/review_queue.py`).
- **Elevation of Privilege** — listing and resolving reviews require
  INTERNAL capability outright.

### 4.6 Client UI (`ui/*`, static pages served by `api/main.py`)

- **Spoofing / Session handling** — there is no server-side session;
  the browser holds the API key in `sessionStorage` only, validated on
  every call via `GET /api/v1/whoami`.
- **Tampering** — review/logs/policy/benchmarks pages independently
  re-check capability server-side on every API call.

## 5. Cross-cutting controls

- **Rate limiting + tenant suspension**: Enforced across endpoints via
  `_reject_suspended_and_rate_limit`. Supported via distributed Redis
  token bucket (`RedisRateLimiter`) with fallback to local in-memory.
- **Metrics/observability**: `gatekeeper_policy_changes_total`, `gateway_call_total`,
  `rate_limited_total` provide high-resolution security observability.
- **Health-check reliability**: `/health` checks `ollama_judge_breaker._opened_at`
  cached state without making fresh blocking requests.
- **Least privilege in container images**: `Dockerfile.api` runs under non-root
  user `gatekeeper` with pre-chowned `/root/.cache/huggingface` and `/app/audit`.

## 6. MCP Transports (stdio & HTTP/SSE) — Trust Models

### 6.1 MCP stdio Transport (`core/mcp_server.py`)
`core/mcp_server.py` runs over stdio with no per-call authentication:
**the process itself is the security boundary**. Bounded via
`readline(MAX_LINE_BYTES + 1)` (1MB cap) with a clean protocol resync loop.

### 6.2 MCP HTTP/SSE Transport (`core/mcp_http_server.py`)
For networked agent environments, `core/mcp_http_server.py` provides:
- **Per-request authentication**: Resolves `Authorization: Bearer <key>` or `X-API-Key` to verify caller identity dynamically.
- **Tenant isolation & suspension**: Suspended tenants receive `403 Forbidden`.
- **Rate limiting**: Requests are throttled per tenant SLA (`429 Too Many Requests`).
- **Payload bounding**: Requests exceeding 100KB are rejected with `413 Payload Too Large`.
- **Session-relayed SSE delivery**: JSON-RPC calls posted to `/mcp/messages?sessionId=<uuid>` enqueue responses to the active session's stream (`202 Accepted`).

## 7. Findings register (this project's own audit trail)

| # | Finding | Class | Status |
|---|---|---|---|
| 1 | `POST /api/v1/policy/rollback` — arbitrary file read via unsanitized `version_name` | Path traversal | Fixed, 4 tests |
| 2 | 16 endpoints (all Phase 7 + review/cache-flush) had no rate limiting or suspension check | Resource exhaustion | Fixed, 21 tests |
| 3 | `ToolCallRequest.arguments`, `GatewayChatRequest.model`/`provider`, `PolicyContentRequest.content`/`version` had no size bound | Resource exhaustion | Fixed, 30 tests combined |
| 4 | `/health` made an uncached blocking call to Ollama every request, compounding under load | Availability / cascading failure | Fixed, 3 tests, real load-test evidence |
| 5 | `core/mcp_server.py`'s stdio loop buffered unbounded line length | Resource exhaustion (low severity) | Fixed, 3 tests |
| 6 | README presented a superseded UI prototype as current | Documentation drift | Fixed |
| 7 | `docker-compose.yml`'s `gatekeeper-ui` service built the superseded prototype | Deployment drift | Fixed — service removed |
| 8 | `Dockerfile.api` ran as root; `/app/audit` volume mount point defaulted to root | Privilege / least-privilege | Fixed, non-root user `gatekeeper` |
| 9 | `core.activity.get_recent_activity` retry logic re-scanned entire log from scratch | Availability / resource exhaustion | Fixed via resume-based single-pass scan (`pos` + `carry`) |
| 10 | `semantic_judge`/`output_judge` fallback prompt injection risk | Prompt injection | Fixed with delimiter tags & framing |

## 8. Known limitations & Operational Boundaries

- **Distributed vs Process-Local Rate Limiting & Token Quotas**
  (`core/rate_limit.py`, `core/token_quota.py`) — When `REDIS_URL` is set,
  `RedisRateLimiter` and `RedisTokenQuotaTracker` enforce global limits across
  all replicas atomically via Redis Lua scripts. When running without Redis,
  instances fall back to process-local in-memory trackers ($N \times$ limit).
- **Rate limiter bucket registry is LRU-capped in fallback mode**
  (`RATE_LIMIT_MAX_TRACKED`) — bounded memory is prioritized over unbounded
  dictionary growth during local fallback.
- **Circuit breakers stay process-local** (`core/circuit_breaker.py`) —
  Circuit breaker state for outbound LLM judges remains process-local.
- **Activity log file tail scanning** (`core/activity.py`) — Resume-based
  reverse scanning eliminates redundant reads and re-parsing, with a hard 20MB
  ceiling (`MAX_BYTES_SCANNED`).

## 9. Explicitly deferred (not part of Phases 1-8)

Per `docs/ROADMAP_V2.md`'s "Explicitly deferred" section: RBAC beyond the
three-tier capability model, an API key management UI, organization
management (hierarchical tenants), dynamic latency-based model routing,
distributed tracing (OpenTelemetry), attack-campaign clustering, compliance
reporting, and automated red-team evaluation.
