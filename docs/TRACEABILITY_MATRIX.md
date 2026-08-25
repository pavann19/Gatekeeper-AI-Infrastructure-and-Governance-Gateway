# Traceability Matrix

Every row maps a requirement or a known threat to the code that implements
its control and the test(s) that verify it. This is the concrete answer to
"how do we know this actually works" for anyone auditing the project — no
row claims coverage without a citation that can be checked.

Cross-references: `docs/THREAT_MODEL.md` (STRIDE analysis + findings
register), `docs/OWASP_COMPLIANCE.md` (standards mapping), `docs/ROADMAP_V2.md`
(phase-by-phase narrative with before/after evidence for every fix).

| Requirement / Threat | Control (code) | Verified by (tests) | Status |
|---|---|---|---|
| Capability must come from a verified credential, never a client assertion | `core/auth.py::resolve_principal` | `tests/test_auth.py`, `tests/test_auth_edge_cases.py` (44 tests), `tests/test_endpoint_auth_matrix.py` (81 tests) | ✅ |
| Concurrent key lookups must never see a partially-loaded key store | `core/auth.py::KeyStore.load()` (atomic dict swap + lock) | `tests/test_keystore_concurrency.py`; live load test `_evidence/perf_slo_whoami.json` (0/300 spurious 401s post-fix, was 12/300 pre-fix) | ✅ (fixed this session) |
| Policy rollback must not allow arbitrary file read via path traversal | `core/policy_versioning.py::rollback_to` (`os.path.basename` guard) | `tests/test_policy_versioning.py`, live pentest `path_traversal` category (4/4) | ✅ |
| Every state-changing/sensitive endpoint enforces rate limiting and tenant suspension | `api/main.py::_enforce_rate_limit`, `_reject_suspended_and_rate_limit` | `tests/test_rate_limit.py`, `tests/test_rate_limit_edge_cases.py` (23 tests), live pentest `rate_limiting`/`tenant_suspension` categories | ✅ |
| Rate limiter never allows N+1 requests when N is the limit | `core/rate_limit.py` (token bucket) | `tests/test_rate_limit_edge_cases.py::test_nth_request_allowed_n_plus_1th_rejected_same_window`; mutation-tested (boundary flip killed by 14 tests) | ✅ |
| Every request field with unbounded size has an explicit cap | `api/schemas.py` (`max_length=...` on every request field) | `tests/test_schemas.py` (79 tests), `tests/test_openapi_contract.py::test_documented_size_bounds_match_enforced_validation` | ✅ |
| INTERNAL-only endpoints reject non-INTERNAL callers, including anonymous | `api/main.py::_require_internal` and inline capability checks | `tests/test_endpoint_auth_matrix.py`, live pentest `auth_bypass` category | ✅ |
| SSRF: the real network-calling tool must reject internal/private/reserved targets | `core/real_tools.py::_resolved_addresses_are_all_public`, hostname allowlist | `tests/test_real_tools.py`, `tests/test_real_tools_edge_cases.py` (17 tests, incl. DNS-rebinding + IPv6 loopback); live pentest `ssrf` category (5/5 real targets rejected post-fix) | ✅ (registration gap fixed this session, see below) |
| The real (non-sandboxed) tool must actually be reachable when a deployment enables it | `api/main.py::warm_models` calling `core.real_tools.register_real_tools()` gated by `REGISTER_REAL_TOOLS` | `tests/test_startup_tool_registration.py` (5 tests, including the WARM_MODELS_ON_STARTUP=False interaction) | ✅ (was completely unreachable in every deployment before this session) |
| PII must be redacted before caching, logging, or judge escalation | `core/privacy.py::redact_pii`; `core/risk.py::assess_risk`'s `raw_prompt` boundary | `tests/test_privacy.py`, `tests/test_logger.py` (raw-content-never-leaked assertions), `tests/test_raw_prompt_detector_boundary.py` | ✅ |
| PII redaction must not itself trigger false-positive detection | `core/risk.py::assess_risk`'s `raw_prompt` parameter (local classifiers see original text) | `tests/test_raw_prompt_detector_boundary.py`; live pentest `benign_2` regression, reproduced with two independent prompts before the fix | ✅ (fixed this session) |
| A cached HIGH-risk verdict must never be silently downgraded | `core/risk.py::assess_risk` (`cache_locked_high` branch) | `tests/test_cache_locked_high_never_downgrades.py` — added this session after mutation testing found NO prior test asserted this | ✅ (gap found and closed this session) |
| Secrets in a response must hard-block, never redact-and-continue | `core/output_guardrails.py::assess_output` (secrets check first, before PII) | `tests/test_output_guardrails.py`, `tests/test_secrets_detection.py` + edge cases (26 tests); mutation-tested (block-condition flip killed by 20 tests) | ✅ |
| System-prompt leakage must be a verbatim substring check, not similarity | `core/output_guardrails.py::check_system_prompt_leakage` | `tests/test_output_guardrails.py`, `tests/test_output_guardrails_edge_cases.py` (boundary at exactly `min_run`); live pentest `system_prompt_leak_is_blocked` | ✅ |
| Output grounding/hallucination check must use the correct reference domain | `core/output_guardrails.py::check_semantic_grounding`, gated by `HALLUCINATION_CHECK_ENABLED` | `tests/test_output_guardrails_edge_cases.py` (both enabled- and disabled-by-default paths) | ⚠️ Disabled by default — real gap found live (wrong reference corpus), gated off rather than deleted pending a correct corpus |
| A timeout must return 503, never a fabricated verdict | `api/main.py`'s `_run_bounded`/`_run_gateway_bounded` timeout handling | `tests/test_request_limits.py` | ✅ |
| Prompt injection / jailbreak attempts must be detected across obfuscation styles | `core/risk.py` (symbolic + fusion + judge), `core/normalizer.py` (evasion defeat) | `tests/test_fusion.py`, `tests/test_normalizer.py` (29 tests, adversarial corpus), live pentest 10-prompt real jailbreak corpus (10/10 blocked, 3/3 benign controls correctly allowed) | ✅ |
| Every tenant only sees its own audit/activity data unless INTERNAL | `api/main.py::_resolve_activity_tenant_scope` | `tests/test_activity_endpoint.py`, `tests/test_endpoint_auth_matrix.py`, live browser walkthrough (cross-tenant trace verified live) | ✅ |
| The Docker image must not run as root | `Dockerfile.api` (dedicated non-root user + pre-chowned volume mounts) | Verified against real builds and real named volumes this session, catching a real `/app/audit` mount-ownership bug in the process | ✅ |
| The generated OpenAPI contract must match real enforced behavior | `api/schemas.py` (source of truth FastAPI generates from) | `tests/test_openapi_contract.py` (4 tests: spec validity, endpoint completeness, size-bound conformance, `additionalProperties: false` conformance) | ✅ (added this session) |
| Every OWASP API/LLM Top 10 category has a stated, cited status | N/A (mapping document) | `docs/OWASP_COMPLIANCE.md` | ✅ 15 covered, 3 partial (stated honestly), 1 not applicable |
| Performance under real concurrent load meets a stated SLO | Bounded thread pools, per-tenant rate limits, timeouts | `_evidence/perf_slo_benchmark.json` — real load test, 4 endpoints, zero 5xx/connection errors across all runs | ✅ |
| Distributed rate limiting across multiple instances | `core/rate_limit.py::RedisRateLimiter` (atomic Lua script + token bucket) | `tests/test_redis_rate_limit.py` (22 tests: Lua script accuracy, TTL expiry, fallback on error) | ✅ |
| Distributed daily token quota tracking across instances | `core/token_quota.py::RedisTokenQuotaTracker` (atomic Lua script + midnight TTL) | `tests/test_redis_token_quota.py` (11 tests: Lua increment, midnight rollover, fallback) | ✅ |
| Per-tenant PII redaction and custom NER entity labeling | `core/privacy.py`, `core/tenancy.py` (`privacy_disabled_patterns`, `privacy_ner_labels`) | `tests/test_tenant_privacy.py` (4 tests: regex bypass override, custom NER labels) | ✅ |
| Single-pass reverse tail scanning without duplicate I/O on zero-match queries | `core/activity.py::_tail_raw_lines` (tracks `pos` and `carry` state) | `tests/test_activity.py` (boundary split tests, manual resume contract test) | ✅ |
| Networked MCP HTTP/SSE transport with per-request authentication and session relay | `core/mcp_http_server.py` (`/mcp/sse`, `/mcp/messages`, `/mcp/jsonrpc`) | `tests/test_mcp_http_server.py` (15 tests: auth, session queue relay, 100KB size cap, 429 rate limit, rate-limiter isolation) | ✅ (2 real gaps found in review and fixed — see below) |
| MCP HTTP session-relayed responses must actually reach the SSE stream, not just return 202 | `core/mcp_http_server.py`'s `queue.put()` on dispatch | `tests/test_mcp_http_server.py::test_mcp_sse_session_relay` (queue-level) plus a live end-to-end run against a real running server (real handshake → real session → real POST → real SSE read) | ✅ (was dead code — session queue created but never written to or validated — fixed in review) |
| MCP HTTP transport must not share a rate-limit budget with the main API for the same key | `core/rate_limit.py::mcp_rate_limiter` (distinct singleton, distinct Redis key namespace from `assess_rate_limiter`) | `tests/test_mcp_http_server.py::test_mcp_rate_limiter_is_isolated_from_the_main_api_limiter` | ✅ (found and fixed in review — originally shared `assess_rate_limiter`, a wrong coupling that would starve budgets across traffic classes once Redis-backed) |
| MCP HTTP transport's error responses for malformed/oversized requests must be valid JSON-RPC 2.0 | `core/mcp_http_server.py::_jsonrpc_error` | `tests/test_mcp_http_server.py::test_mcp_malformed_json_returns_400`, `test_mcp_non_dict_json_returns_400`, `test_mcp_oversized_payload_returns_413` | ✅ (found and fixed in review — a first pass returned FastAPI's generic `{"detail": ...}` shape instead of the JSON-RPC envelope) |
| Outbound live LLM provider validation and latency diagnostics | `scripts/exercise_live_providers.py` (`exercise_provider`) | `tests/test_exercise_live_providers.py` (3 tests: response parsing, exception handling) | ✅ |

## Coverage summary

- **1580 automated tests** as of this session (1522 at session start).
- **8 real mutants** manually tested against security-critical functions:
  6 killed cleanly, 1 revealed and closed a real gap, 1 correctly deferred
  to existing coverage by the safety classifier.
- **46/46 live black-box pentest checks** passing (2 real bugs found and
  fixed in the process, not counted as pre-existing passes).
- **4/4 real load-test runs** with zero 5xx/connection errors (1 real
  concurrency bug found and fixed in the process).
- **8 real bugs found and fixed this session** that no prior test had
  caught: PII-redaction-poisons-detectors, output-hallucination-check-
  wrong-corpus, `http.get` never registered, startup tool-registration
  unreachable when warm-up disabled, KeyStore concurrency race, MCP HTTP
  transport's SSE session relay being dead code, MCP HTTP transport
  sharing a rate-limit bucket with the main API, MCP HTTP transport's
  parse/size errors not being valid JSON-RPC.
- **Review discipline note**: the last 3 bugs above were found by
  independently re-verifying another contributor's "done" work against
  this project's own bar (live end-to-end testing, not just reading the
  diff) rather than trusting its own tests and docs at face value — the
  first pass's tests for the MCP session relay asserted only that a 202
  came back, never that the queue was actually written to or that a real
  SSE client would ever see the response.

