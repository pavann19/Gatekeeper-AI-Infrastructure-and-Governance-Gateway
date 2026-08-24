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
(`core/mcp_server.py`) is covered separately (§6) because it has a
genuinely different trust model, not because it was skipped.

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
| PII in transit | Regex + NER pipeline (`core/privacy.py`) | Redacted before logging or forwarding, not merely flagged |

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

### 4.2 Tool/Agent Gateway (`/api/v1/tools/call`, `core/tools.py`)

- **Elevation of Privilege** — a tool call is decided in a fixed order
  (access control → structural validation → risk-based approval),
  cheapest and most decisive first (`core/tools.py::decide_tool_call`'s
  own docstring). A HIGH-risk tool always requires human REVIEW even for
  a caller whose capability already clears access control — "may this
  caller use this tool" and "is this SPECIFIC call safe without a human"
  are deliberately not conflated.
- **Tampering** — `core/real_tools.py`'s `http.get` (the one tool that
  makes a real outbound network call) is the SSRF case study: a
  hostname allowlist alone is insufficient against DNS rebinding, so
  every call also resolves the hostname and rejects it if ANY address is
  private/loopback/link-local/reserved, checked at call time, not once
  at startup. Redirects are never followed. Response size and request
  duration are both capped. Verified live against the classic SSRF
  target `169.254.169.254` and against `localhost`.
- **Denial of Service** — §7 finding #3: `ToolCallRequest.arguments` had
  no size bound at all before Phase 8 (every other free-text field in
  the API did); fixed with a 100KB serialized cap plus per-field
  JSON-Schema `maxLength` support in `validate_arguments`, applied
  concretely to `http.get`'s `url` (2048 chars).

### 4.3 LLM Gateway (`/api/v1/gateway/chat`, `core/llm_providers.py`)

- **Spoofing** — the outbound provider endpoint (`base_url`) is always
  server-configured (`settings.OLLAMA_CHAT_URL` / `OPENAI_BASE_URL` /
  `ANTHROPIC_BASE_URL`); `get_provider(name)` takes no caller-supplied
  URL override, so a caller can select WHICH configured provider to use,
  never WHERE that provider actually points. Confirmed by code review
  (§7) — no injection vector found here.
- **Denial of Service** — §7 finding #3 (continued): `model` and
  `provider`, the two caller-supplied fields forwarded into the outbound
  request, had no length bound before Phase 8. Fixed (200 / 100 chars).
- **Information Disclosure** — token usage is metered per-tenant
  (`core/token_quota.py`) only from what a provider actually reports;
  Ollama reports none, so usage-based quota simply does not apply to it
  rather than estimating a number that would misrepresent actual spend.

### 4.4 Policy Engine and Policy Editor (`/api/v1/policy/*`)

- **Tampering, the confirmed-and-fixed case** — §7 finding #1, the most
  serious this project's Phase 8 audit found: `core.policy_versioning
  .rollback_to` built its restore path with `os.path.join(_versions_dir
  (), version_name)`, safe when the only caller was a trusted local CLI
  operator (who already has filesystem access), and a real
  arbitrary-file-read primitive the moment `POST /api/v1/policy/rollback`
  exposed it to any INTERNAL API key over the network (`os.path.join`
  silently discards its base directory when given an absolute path —
  reproduced live: `os.path.join("/a", "/etc/passwd") ==
  "/etc/passwd"`). Fixed by rejecting any `version_name` that isn't a
  bare filename. 4 regression tests
  (`tests/test_policy_versioning.py`).
- **Elevation of Privilege** — deploy is validated BEFORE it is ever
  written (`validate_policy_file`, refusing outright on any error,
  mirroring the CLI's own `cmd_deploy`), and every deploy/rollback
  snapshots the outgoing policy first — a bad change is exactly as
  undoable as a bad rollback.
- **Denial of Service** — `PolicyContentRequest.content` is capped at
  1MB, `PolicyRollbackRequest.version` at 255 chars (the first instance
  of the pattern §4.2/§4.3 later repeated).

### 4.5 Human Review Queue (`/api/v1/review/*`)

- **Information Disclosure** — a review record stores only a
  `prompt_hash`/argument hash, never raw content, reusing the SAME
  field the audit log's own privacy contract already guarantees
  (`core/review_queue.py`).
- **Elevation of Privilege** — listing and resolving reviews require
  INTERNAL capability outright; reading one's own review status only
  requires authentication, a deliberately lower bar since "is my
  request still pending" is not the sensitive question "what is
  everyone else's pending request."

### 4.6 Client UI (`ui/*`, static pages served by `api/main.py`)

- **Spoofing / Session handling** — there is no server-side session;
  the browser holds the API key in `sessionStorage` only (cleared on tab
  close, never written to disk), and every page call re-validates it via
  `GET /api/v1/whoami` against the real KeyStore before trusting
  anything client-side.
- **Tampering** — the login page's redirect target is INTERNAL-vs-other
  aware but the review/logs/policy/benchmarks pages independently
  re-check capability server-side on every call; a forged client-side
  state (e.g. editing `sessionStorage` to claim INTERNAL) gets a real
  403 from the API, not a UI that merely hides itself.

## 5. Cross-cutting controls (apply to every endpoint above)

- **Rate limiting + tenant suspension**, §7 finding #2: originally only
  the four original endpoints (assess/assess_output/gateway_chat/
  tools_call) enforced either. Every endpoint added since Phase 7 — and
  the pre-existing review/cache-flush endpoints — had NEITHER. Closed
  in one shared helper (`api/main.py::_reject_suspended_and_rate_limit`)
  across all 16 affected endpoints rather than fixed ad hoc. Explicitly
  NOT framed as anti-brute-force (API keys are 256-bit random tokens,
  `secrets.token_urlsafe(32)` — online guessing is infeasible regardless
  of throttling); the real justification is resource exhaustion, since
  several of these endpoints do real per-call work (a byte-bounded
  audit-log scan, a policy-file write-then-validate-then-delete).
- **Metrics/observability** — `gatekeeper_policy_changes_total` tracks
  policy deploy/rollback outcomes as a first-class security event, not
  just a generic HTTP status code, matching the precedent
  `gateway_call_total` already set.
- **Health-check reliability**, a real load-test finding (`docs/
  ROADMAP_V2.md`'s Phase 8 load-testing entry): `/health` made a fresh,
  uncached, blocking network call to Ollama on EVERY request, with no
  reuse of `core.circuit_breaker.ollama_judge_breaker`'s already-tracked
  state from the real judge path. Under concurrent load with Ollama
  down, this degraded `/health` itself to multi-second latency — the
  worst possible failure mode for a liveness/readiness probe, since an
  orchestrator could kill a healthy API process over a slow EXTERNAL
  dependency. Fixed by consulting the breaker's cached state first,
  reading `_opened_at` directly (never calling the side-effecting
  `is_open()`, which would have silently consumed the breaker's one-shot
  half-open probe from a health check that never actually attempts a
  judge call).
- **Least privilege in the container image**, §7 finding #8:
  `Dockerfile.api` ran as root with no `USER` directive. Fixed with a
  dedicated non-root user, verified against a real build and real named
  volumes (not assumed from documentation) — including catching and
  fixing a second real bug in the same pass, `/app/audit`'s volume mount
  point defaulting to root ownership because nothing pre-existed there
  in the image for Docker to inherit permissions from.

## 6. The MCP stdio transport — a different trust model, stated explicitly

`core/mcp_server.py`'s own docstring is explicit: an MCP client connects
over stdio with no request-level authentication at all, and this server
therefore runs at ONE fixed capability for its entire process lifetime.
**The process itself is the security boundary, not each individual
message within it** — the same shape every real-world local MCP server
already has. This is why the module is analyzed separately: applying
the HTTP endpoints' per-request authorization model to it would be a
category error, not extra rigor.

What Phase 8 DID still harden here, because it costs nothing regardless
of trust level: `for line in stream` buffered an entire line into memory
before yielding it, an unbounded read in principle even for a trusted
client that misbehaves by accident. Replaced with
`readline(MAX_LINE_BYTES + 1)` (1MB cap) plus a resync loop so an
oversized line is reported as a clean protocol error and does not
desynchronize the next real message from being parsed correctly. 3 new
tests (`tests/test_mcp_server.py`).

**Explicitly out of scope, not an oversight**: per-caller authorization
over MCP would require a different transport (MCP's HTTP/SSE transport,
with its own auth story) — `core/mcp_compat.py`'s own docstring already
states this was deferred, not forgotten.

## 7. Findings register (this project's own audit trail)

Every finding below was found by actually reading the code or actually
running a test against a real instance — never assumed. Full narrative
for each lives in `docs/ROADMAP_V2.md`'s Phase 8 section; this is the
severity-ranked index.

| # | Finding | Class | Status |
|---|---|---|---|
| 1 | `POST /api/v1/policy/rollback` — arbitrary file read via unsanitized `version_name` | Path traversal | Fixed, 4 tests |
| 2 | 16 endpoints (all Phase 7 + review/cache-flush) had no rate limiting or suspension check | Resource exhaustion | Fixed, 21 tests |
| 3 | `ToolCallRequest.arguments`, `GatewayChatRequest.model`/`provider`, `PolicyContentRequest.content`/`version` had no size bound | Resource exhaustion | Fixed, 30 tests combined |
| 4 | `/health` made an uncached blocking call to Ollama every request, compounding under load | Availability / cascading failure | Fixed, 3 tests, real load-test evidence |
| 5 | `core/mcp_server.py`'s stdio loop buffered unbounded line length | Resource exhaustion (low severity — trusted-process model) | Fixed, 3 tests |
| 6 | README presented a superseded, pipeline-bypassing UI prototype as current | Documentation drift (operational risk: wrong UI deployed) | Fixed |
| 7 | `docker-compose.yml`'s `gatekeeper-ui` service built the same superseded prototype | Documentation/deployment drift | Fixed — service, `Dockerfile.ui`, and `ui/web_app.py` removed outright |
| 8 | `Dockerfile.api` ran as root; `/app/audit` volume mount point defaulted to root ownership once non-root was introduced | Privilege / least-privilege | Fixed, verified live with real named volumes; full end-to-end healthy-boot not confirmed under this test run's memory constraints (see §5) |

## 8. Known limitations (accepted, not hidden)

Each of these is already stated in the relevant module's own docstring;
listed here so a reviewer doesn't have to go find them individually.

- **Process-local rate limiter and circuit breakers**
  (`core/rate_limit.py`, `core/circuit_breaker.py`) — a multi-worker
  deployment gets N× the configured limit, since each worker enforces
  independently. A distributed version needs shared state (the Redis
  instance `core/cache_backend.py` already introduces is the stated
  natural home).
- **Rate limiter bucket registry is LRU-capped**
  (`RATE_LIMIT_MAX_TRACKED`) — bounded memory was judged more important
  than perfect per-identity accounting; a caller cycling through more
  distinct identities than the cap can evict its own bucket and reset
  its budget. Anonymous callers are keyed by peer address specifically
  so this isn't free for them.
- **No distributed token quota** (`core/token_quota.py`) — same
  process-local caveat, same stated future home.
- **MCP has no per-caller authorization** (§6) — a deliberate scope
  boundary, not a gap.
- **The activity/logs/trace endpoints' real scalability under load** —
  a real load test (`docs/ROADMAP_V2.md`) showed severe latency
  degradation under concurrency that was root-caused to `/health`'s
  Ollama probe, not `core/activity.py`'s file scan — but the test
  environment had other load-generating processes running throughout,
  so this is recorded as an open question, not a closed one. Revisit
  with a dedicated, otherwise-idle load-test environment before
  concluding the audit-log tail-read design needs no further work.

## 9. Explicitly deferred (not part of Phases 1-8)

Per `docs/ROADMAP_V2.md`'s own "Explicitly deferred (280-400h tier)"
section: full MCP HTTP/SSE transport with per-call auth, RBAC beyond the
three-tier capability model, an API key management UI, organization
management, distributed rate limiting/quotas, model routing/fallback
beyond the current provider-chain fallback, distributed tracing, and
automated red-team evaluation. Not started, not scheduled, and not
claimed as covered by anything in this document.
