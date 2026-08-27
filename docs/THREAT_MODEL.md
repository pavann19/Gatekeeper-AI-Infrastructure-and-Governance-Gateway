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
(`core/mcp_server.py`) and the MCP HTTP/SSE transport
(`core/mcp_http_server.py`) are covered separately (§6) because each has
a genuinely different trust model, not because either was skipped.

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
| PII in transit | Regex + NER pipeline (`core/privacy.py`) | Redacted before logging or forwarding, not merely flagged. Per-tenant overrides (`privacy_disabled_patterns`, `privacy_ner_labels` in `core/tenancy.py`) are themselves operator-provisioned config, not caller-controlled — a tenant cannot loosen its own redaction via any API call, only an operator editing `tenants.json` can. |

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
  the classifier deciding SAFE/DANGEROUS. Lower real-world severity than
  it first appears: the actually-validated, default path
  (`_judge_via_llama_guard`) sends untrusted content cleanly via the
  chat API's `user` role, never string-concatenated, and Llama Guard is
  a fine-tuned classifier that ignores embedded instructions entirely by
  construction — this only affects a non-default configuration.
  Mitigated with delimiter tags and explicit "treat as data, not
  instructions" framing regardless, since the fix costs nothing on the
  path that's actually used.
- **Tampering (redaction poisoning the classifiers it feeds)**, §7
  finding #11: `[REDACTED:PERSON]`-style placeholder tokens are
  themselves out-of-distribution for text classifiers trained on natural
  language, so a benign redacted prompt could get falsely flagged by the
  fusion detectors purely because of the redaction artifact, not the
  underlying content. Fixed by giving `assess_risk` a `raw_prompt`
  parameter: the local, non-persisted, in-process classifiers see the
  original text, while the embedding, the persisted semantic cache, and
  the judge escalation all stay on the redacted text exactly as before —
  a narrow, deliberate boundary, not a blanket rollback of the redaction
  discipline.

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
- **Elevation of Privilege (registration gap)**, §7 finding #12:
  `core.real_tools.register_real_tools()` — the function that wires up
  `http.get`, the only real tool this project ships, with its
  already-tested SSRF protection — was fully built and unit-tested but
  never actually called anywhere. Not a vulnerability in the SSRF
  protection itself (it was never reachable to be exploited), but a real
  gap between "this control exists in the codebase" and "this control
  is enforced in any real deployment." Fixed by adding
  `REGISTER_REAL_TOOLS` (default off, same opt-in contract as
  `REGISTER_DEMO_TOOLS`) and wiring it into `api/main.py::warm_models`'s
  startup handler — which surfaced a second, independent bug in the
  same function: tool registration used to sit behind an early `return`
  gated on `WARM_MODELS_ON_STARTUP`, so a deployment running with model
  warm-up disabled silently never registered ANY tools either, even with
  `REGISTER_DEMO_TOOLS` explicitly set. Restructured so tool
  registration always runs regardless of the warm-up setting.

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
  Now enforced consistently across replicas when `REDIS_URL` is
  configured (`RedisTokenQuotaTracker`, §5/§8).

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
  audit-log scan, a policy-file write-then-validate-then-delete). Now
  backed by `RedisRateLimiter` (atomic Lua-script token bucket, §8) when
  `REDIS_URL` is configured, so the limit is enforced exactly across
  replicas rather than N× multiplied by worker count — with a dedicated
  bucket namespace (`assess_rate_limiter`, distinct from the MCP HTTP
  transport's own `mcp_rate_limiter`, §6.2) so one traffic class can
  never starve another's budget.
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
- **Concurrent key-store loads must never serve a half-populated key
  set**, found by a real load test (not a unit test): `core.auth
  .KeyStore.load()` set `self._loaded = True` before finishing
  repopulating `self._keys`, so a concurrent `lookup()` from another
  thread (real concurrency — FastAPI dispatches sync endpoints like
  `whoami` to a thread pool) could observe an empty or half-built dict
  during the window right after a fresh load or a forced reload.
  Measured live: 12/300 (4%) spurious `401 Unauthorized` responses for a
  demonstrably valid API key under concurrent `/api/v1/whoami` load.
  Fixed by building the new key dict in a local variable and swapping it
  into `self._keys` in one atomic assignment only after it's fully
  populated, with a lock serializing concurrent loads
  (`tests/test_keystore_concurrency.py`). Re-run of the identical load
  test after the fix: 0/300 spurious 401s.

## 6. MCP transports — two different trust models, both stated explicitly

### 6.1 The MCP stdio transport (`core/mcp_server.py`)

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

### 6.2 The MCP HTTP/SSE transport (`core/mcp_http_server.py`)

Unlike stdio, this transport IS network-facing (its own standalone
FastAPI app, its own process/port via `scripts/run_mcp_http_server.py`,
never mounted into the main API) and therefore gets the full per-request
STRIDE treatment stdio deliberately does not.

- **Spoofing** — every request resolves its principal server-side via
  the SAME `core.auth.resolve_principal` the main API trusts
  (`Authorization: Bearer` or `X-API-Key`, checked in that order); no
  caller-asserted identity is trusted, identical discipline to §1.
- **Elevation of Privilege** — a suspended tenant is rejected (403)
  before any tool dispatch happens, and dispatch itself routes through
  the SAME `core.mcp_server.handle_request`/`core.tools` capability
  checks the stdio transport uses — there is no separate, potentially
  weaker authorization path for the networked transport.
- **Denial of Service** — request bodies are capped at 100KB
  (`MAX_PAYLOAD_BYTES`), read and size-checked BEFORE JSON parsing is
  even attempted (cheapest check first). Rate limited per authenticated
  key (or per source IP for anonymous callers) via a dedicated
  `mcp_rate_limiter` singleton (`core/rate_limit.py`).
- **Elevation of Privilege / Denial of Service (a real bug found and
  fixed in this session's review, not by the original implementation)**,
  §7 finding #13: the MCP HTTP server originally imported and reused
  `api/main.py`'s own `assess_rate_limiter` singleton, keyed identically
  by `key:{key_id}`. In the Redis-backed configuration — the entire
  point of the distributed rate limiter existing — this would mean a
  caller's MCP traffic and their main-API traffic drain the SAME shared
  budget, exactly the "wrong coupling between traffic classes"
  `api/main.py`'s own `_gateway_pool`/`_assess_pool` separation already
  guards against for an analogous reason. Fixed by giving the MCP server
  its own `mcp_rate_limiter` singleton with a distinct Redis key
  namespace, confirmed as genuinely separate objects
  (`tests/test_mcp_http_server.py::test_mcp_rate_limiter_is_isolated_from_the_main_api_limiter`).
- **Tampering (protocol conformance), §7 finding #14**: a first pass at
  adding the 100KB size cap and JSON validation above returned parse/
  shape errors as FastAPI's generic `{"detail": "..."}` body instead of
  a JSON-RPC 2.0 error envelope (`{"jsonrpc": "2.0", "id": null,
  "error": {"code": ..., "message": ...}}`) — a real protocol-conformance
  regression a strict MCP client parsing every response as JSON-RPC
  first would fail to read correctly. Fixed by routing parse/shape
  failures through a dedicated `_jsonrpc_error` helper that returns the
  correct envelope with the correct reserved codes (`-32700` Parse
  error, `-32600` Invalid Request, `-32000` payload-too-large as an
  implementation-defined server error), verified live end-to-end against
  a real running server.
- **Session relay integrity**, §7 finding #15: the SSE session queue
  (`GET /mcp/sse` opens one, advertises `/mcp/messages?sessionId=<uuid>`)
  was originally created but never written to or validated — any
  `sessionId`, including one that was never opened, was silently
  accepted and the response returned directly in the POST's HTTP body
  instead of being relayed over the open SSE stream, which is what the
  MCP HTTP+SSE transport specification requires. Fixed: an unknown/
  expired `sessionId` now returns a real 404, and a valid one enqueues
  the JSON-RPC response into that session's queue for delivery as an
  `event: message` SSE frame, with the POST itself returning 202
  Accepted per spec. Verified live end-to-end against a real running
  server (real handshake → real session id → real POST → real SSE
  stream read), not just against a unit test that injects a queue
  directly.
- **Least surprise / operational default**: `scripts
  /run_mcp_http_server.py --demo-tools` defaults OFF (`action=
  "store_true"`), matching this project's established "opt-in, not
  automatic" convention for demo tooling (`REGISTER_DEMO_TOOLS` in the
  main API) rather than the CLI's own first draft, which defaulted it on.

**Explicitly out of scope, not an oversight, for BOTH transports**: full
RBAC beyond the three-tier capability model, and organization-level
tenant hierarchy, remain deferred (§9) regardless of transport.

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
| 9 | `core.activity.get_recent_activity`'s retry logic re-scanned the entire audit log from scratch (several times) when a filter matched zero entries | Availability / resource exhaustion (~45x measured latency cliff) | Fixed, capped at 2 scan attempts instead of 5-6; further improved to a resume-based single-pass scan (`pos`/`carry` state) that never re-reads bytes an earlier pass already covered; 47-86% latency reduction measured live, regression tests including a full-file-reverse-iteration cross-check |
| 10 | `semantic_judge`/`output_judge`'s non-Llama-Guard fallback path concatenated untrusted content directly into the judge's instruction prompt with no delimiter | Prompt injection (low severity — not the validated/default model path) | Fixed with delimiter tags + explicit data-not-instructions framing; 3 new tests |
| 11 | PII redaction's `[REDACTED:PERSON]`-style placeholder tokens are themselves out-of-distribution for the fusion detectors, causing false-positive blocking on entirely benign prompts | False positive / availability (detection over-blocking) | Fixed via `assess_risk`'s `raw_prompt` parameter (local classifiers only); reproduced live with two independent prompts before the fix |
| 12 | `core.real_tools.register_real_tools()` was fully built and unit-tested but never called anywhere — `http.get` was unreachable in every real deployment; a second, independent bug in the same startup function made tool registration unreachable whenever `WARM_MODELS_ON_STARTUP=False` | Elevation of privilege (missing control) / dead code | Fixed: `REGISTER_REAL_TOOLS` setting (default off) wired into startup; registration decoupled from the warm-up gate; 5 new tests |
| 13 | MCP HTTP transport shared `api/main.py`'s `assess_rate_limiter` singleton, so MCP and main-API traffic for the same key drained one budget once Redis-backed | Resource exhaustion / wrong coupling between traffic classes | Fixed: dedicated `mcp_rate_limiter` singleton with its own Redis key namespace; isolation confirmed by test |
| 14 | MCP HTTP transport's payload-size/parse-error handling returned FastAPI's generic error shape instead of a JSON-RPC 2.0 error envelope | Protocol conformance | Fixed via a dedicated `_jsonrpc_error` helper; verified live |
| 15 | MCP HTTP transport's SSE session queue was created but never written to or validated — any `sessionId` was accepted and responses were never actually relayed over the SSE stream | Protocol conformance / broken feature | Fixed: session validation (404 on unknown), real queue relay, 202 Accepted per spec; verified live end-to-end against a real running server |
| 16 | `output_judge()` returning `JUDGE_OFFLINE` when the primary judge (Ollama/Llama Guard) is unreachable fell through `assess_output`'s `if verdict == "DANGEROUS"` check to silent ALLOW — a fail-**open** on exactly the outage a real deployment will eventually hit (Ollama restarted, OOM'd, host rebooted), and the window an attacker motivated to force it would want | Improper error handling / fail-open on a security control | Fixed: `output_judge()` now falls back to `toxic_bert` (`core.semantic_judge._fallback_output_judge`) — already loaded and warmed at process startup for the INPUT-side fusion ensemble, so no cold-start cost at the moment it's needed — before ever returning the bare `JUDGE_OFFLINE` sentinel; gated by `OUTPUT_JUDGE_FALLBACK_ENABLED` (default on). True fail-open is now reached only if BOTH the primary judge and this fallback are down. Verified live against the real model: 71ms warm-call latency, correctly classified both a toxic and a benign response; 7 new regression tests (`tests/test_semantic_judge.py`).

## 8. Known limitations (accepted, not hidden)

Each of these is already stated in the relevant module's own docstring;
listed here so a reviewer doesn't have to go find them individually.

- **Rate limiter and token quota, distributed vs. process-local** — both
  `core/rate_limit.py` and `core/token_quota.py` now support a
  Redis-backed distributed mode (`RedisRateLimiter`,
  `RedisTokenQuotaTracker`, atomic Lua scripts, server-clock-based
  refill, TTL auto-expiry) selected automatically when `REDIS_URL` is
  set and reachable, with a transparent fallback to the original
  process-local `LocalRateLimiter`/`LocalTokenQuotaTracker` when it is
  not (or during a transient Redis outage). Without `REDIS_URL`
  configured, the original limitation still applies exactly as before:
  a multi-worker deployment gets N× the configured limit, since each
  worker enforces independently. `docs/DEPLOYMENT_RUNBOOK.md` records
  the release-layer decision for which mode a given deployment runs in
  — this is now a configuration choice, not an unconditional gap.
  RESOLVED, not just revisited: the Redis-backed paths were first
  verified against the real `redis-py` client API and a stateful
  Lua-equivalent test simulation, then against a real live Redis server
  (`redis:7-alpine`) under real concurrent access — 200 concurrent
  threads against a 50-token bucket never overspent capacity (the Lua
  script's atomicity genuinely holds server-side), 100 concurrent token
  increments summed to exactly the correct total, and a real
  Redis-becomes-unreachable case fell back cleanly. Then verified the
  actual production scenario directly: two independent `uvicorn`
  processes plus a separate MCP HTTP server process, all against the
  same real Redis — a rate-limit burst against one process was
  immediately visible as exhausted on the other (genuinely different OS
  processes, not threads), while the MCP server's own rate limiter
  (same key, same Redis, same instant) stayed correctly unaffected.
  `scripts/live_redis_verification.py`,
  `_evidence/live_redis_verification_results.json`.
- **Rate limiter bucket registry is LRU-capped in local-fallback mode**
  (`RATE_LIMIT_MAX_TRACKED`) — bounded memory was judged more important
  than perfect per-identity accounting; a caller cycling through more
  distinct identities than the cap can evict its own bucket and reset
  its budget. Anonymous callers are keyed by peer address specifically
  so this isn't free for them. The Redis-backed mode does not have this
  limitation (Redis TTL evicts idle keys, not an LRU cap on tracked
  identity count).
- **Circuit breakers stay process-local regardless of Redis
  configuration** (`core/circuit_breaker.py`) — unlike rate limiting and
  token quotas, judge-backend health tracking has no distributed
  counterpart yet; each replica discovers an Ollama outage independently.
- **MCP stdio has no per-caller authorization** (§6.1) — a deliberate
  scope boundary for that transport specifically, not a gap; the MCP
  HTTP/SSE transport (§6.2) DOES have per-request authorization, so this
  limitation no longer applies to MCP as a whole, only to the stdio
  transport by design.
- **The activity/logs/trace endpoints' real scalability under
  zero-match queries** — RESOLVED, not just revisited: a follow-up load
  test in a quiet environment isolated a second, independent bug in
  `core.activity.get_recent_activity` (its retry-window growth strategy
  re-scanned the entire audit log from scratch when a filter matched
  nothing at all — a real, ~45x latency cliff, confirmed and fixed; see
  `docs/ROADMAP_V2.md`'s Phase 8 load-testing entry for the full
  before/after numbers), then further improved to a resume-based
  single-pass scan that tracks byte offset and carry state across the
  tail read so a second pass never re-reads bytes the first pass already
  covered. What remains a genuine, stated limitation rather than a
  closed question: a full-`MAX_BYTES_SCANNED` (20MB) scan for a query
  that truly matches nothing is still real work under concurrency —
  closing that further would need an index, a larger change than this
  pass attempted.
- **No cross-tenant embedding partitioning in the semantic cache/vector
  stores** (`core/cache.py`, `core/vector_store.py`) — a fuzzy-match hit
  from one tenant's traffic can influence another's cached risk verdict
  for a similar prompt. The underlying raw text never leaks (only the
  verdict/similarity can), and this is an accepted architectural
  simplification, not yet fully mitigated.

## 9. Explicitly deferred (not part of Phases 1-8)

Per `docs/ROADMAP_V2.md`'s own "Explicitly deferred" section: RBAC
beyond the three-tier capability model, an API key management UI,
organization management (hierarchical tenants), dynamic latency-based
model routing beyond the current provider-chain fallback, distributed
tracing (OpenTelemetry), attack-campaign detection/clustering,
compliance reporting (SOC2/ISO27001 export), and automated red-team
evaluation. Not started, not scheduled, and not claimed as covered by
anything in this document. (Full MCP HTTP/SSE transport support was
previously listed here as deferred — it is now built, §6.2, and removed
from this list accordingly.)
