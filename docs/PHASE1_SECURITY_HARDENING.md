# GATEKEEPER PHASE 1 COMPLETE

### Release decision

PHASE 1 BLOCKED

Implementation is reviewable in the working tree. Canonical model acceptance, the full replay gate and the type gate are not green. No commit, push, publication, history rewrite or GitHub settings change was performed. Base remains main at 6b1cf091c6717b494d9e6bce123e941d379419d2.

### Security fixes

1. Tenant policy: distinguish optional unconfigured policy from configured valid/unavailable policy. Missing or corrupt configured stores deny; malformed entries deny; immutable snapshots under a lock prevent concurrent reload from publishing a default-active state.
2. Output: only explicit SAFE permits. DANGEROUS, unavailable judges, invalid responses, errors and timeouts block without clean output; assessment status remains explicit across standalone, combined and gateway paths.
3. Cache: execute current symbolic veto before reuse. Bind raw and cleaned identity, tenant and security/evaluation version. LOW/MEDIUM are reassessed; only HIGH is reusable. Retain unexpired HIGH atomically within the same scope across local, Redis-script and background writes. TTL expiry, version changes, capacity eviction and operator flushing are not permanent HIGH retention guarantees.
4. Audit: explicit append/flush/fsync or configured SQLite authority must acknowledge persistence. Failures are observable and deny success even with the legacy opt-out. Persist tool execution intent before the handler, then completion/failure. A completion-storage failure cannot undo an already completed side effect, but never returns successful execution.
5. Review: authenticated ownership checks; missing and foreign records return the same opaque 404. Explicit trusted INTERNAL authority retains cross-tenant read/resolve.
6. MCP: authenticate and enforce suspension/rate admission at handshake and POST; bind sessions to verified credentials/principal; cap sessions, streamed request bodies, queues, pending dispatch and workers. Blocking work runs off the event loop; timed-out work retains its slot until completion. Capability and HIGH REVIEW semantics remain enforced.
7. SSRF: validate every DNS result and connect to the chosen validated address. Preserve original Host and TLS hostname verification. Ignore environment proxies and do not follow redirects. Support IPv4/IPv6. A separate process enforces the total DNS/connect/TLS/stream deadline and is reaped after timeout.
8. Supporting correctness: required NER unavailable denies; own gk_ token boundaries include trailing dash/underscore; sanitize provider, tool and stdio failure text.
9. Shared helpers: narrowly share admission, ownership and audit-failure handling without a broad architecture rewrite.
10. Negative controls: original 20 security regressions failed before fixes and passed after. Separate stdio leak regression also failed before its fix. Three independent mutations were killed.
11. Verification: execute affected suites, CI selection, full slow selection, coverage, lint and type checks; preserve failed logs and unavailable infrastructure labels below.
12. Release discipline: retain frozen fixtures, benchmark claims and original backups; raw logs remain outside the public repository.

### Files changed

- `api/main.py` — Shared admission, tenant-scoped assessment, authoritative audit failure responses, bounded tools, private review ownership.
- `core/admission.py` — Shared authenticated admission and tenant ownership helpers.
- `core/audit_store.py` — Durable SQLite commits and compatible output/tool audit columns.
- `core/cache.py` — Raw/clean/tenant/security-version identity; HIGH-only reuse and monotonic retention.
- `core/cache_backend.py` — Atomic local persistence and Redis Lua HIGH retention.
- `core/config.py` — Corrected security behavior documentation.
- `core/llm_providers.py` — Sanitized provider failure details.
- `core/logger.py` — Explicit authoritative persistence, observable failure, output status and tool phases.
- `core/mcp_http_server.py` — Authenticated owned SSE sessions, bounded ingress/queues/workers, off-loop dispatch.
- `core/mcp_server.py` — Sanitized stdio dispatch failure diagnostics.
- `core/output_guardrails.py` — Only explicit SAFE permits; unavailable/error/timeout/invalid output blocks.
- `core/privacy.py` — Required NER unavailable fails closed.
- `core/real_tools.py` — Validated address pinning, original TLS/Host identity, process-enforced total deadline.
- `core/risk.py` — Current symbolic veto precedes cache; background escalation retains original scope.
- `core/secrets_detection.py` — Own API key detection including trailing dash/underscore.
- `core/tenancy.py` — Atomic configured/unconfigured/unavailable tenant state.
- `core/tools.py` — Durable intent before side effects and completion/failure afterward; sanitized untrusted fields.
- `docs/OWASP_COMPLIANCE.md` — Corrected directly affected output fallback statement.
- `docs/PHASE1_SECURITY_HARDENING.md` — This report and exact change manifest.
- `docs/THREAT_MODEL.md` — Corrected directly affected output fallback statement.

### Tests added/changed

- `tests/test_audit_store.py` — Authoritative storage failure propagation.
- `tests/test_cache.py` — HIGH-only reuse and eviction assertions.
- `tests/test_cache_backend.py` — Atomic Redis Lua contract rather than separate SETEX.
- `tests/test_cache_backend_edge_cases.py` — Redis eval serialization failures.
- `tests/test_cache_edge_cases.py` — Versioned keys, permissive misses, concurrent HIGH and fuzzy LRU.
- `tests/test_fault_injection.py` — Audit failure remains 503 despite legacy opt-out.
- `tests/test_llama_guard_async.py` — Background escalation preserves original raw/tenant scope.
- `tests/test_logger.py` — Explicit persistence replaces diagnostic handler authority.
- `tests/test_mcp_compat.py` — Intent/completion audit and sanitized errors.
- `tests/test_mcp_http_server.py` — Owned SSE session relay.
- `tests/test_mutants_redis_and_mcp.py` — Redis and owned MCP queue contracts.
- `tests/test_output_guardrails_edge_cases.py` — Assessment status fields and invalid verdict blocking.
- `tests/test_phase1_boundaries.py` — Audit storage faults and tool ordering; MCP auth/ownership/bounds; review isolation; raw/tenant/version cache scope; SSRF pinning; privacy and provider/stdio error sensitivity; output endpoint failures.
- `tests/test_phase1_network_transport.py` — Actual TCP/TLS trusted/untrusted/wrong-host transport, DNS stall and slow-drip total deadline, child process reaping.
- `tests/test_phase1_security_invariants.py` — Tenant corruption/missing/malformed/concurrent reload; output verdict failures; symbolic veto; monotonic local HIGH; own key boundaries.
- `tests/test_prometheus_metrics.py` — Tenant assessment signature and diagnostic observation compatibility.
- `tests/test_real_tools.py` — Socket transport, caps, redirects, allowlists, closure and original scheme cases.
- `tests/test_real_tools_edge_cases.py` — HTTP byte stream failures and resource closure.
- `tests/test_request_limits.py` — Tenant assessment signature with request limits.
- `tests/test_review_endpoints.py` — Authenticated owner review probes.
- `tests/test_tenancy.py` — Missing/invalid configured store denial.
- `tests/test_tenancy_edge_cases.py` — Malformed tenant retained as unavailable.
- `tests/test_tool_audit_events.py` — Allowed intent/completion; blocked/review audit semantics.
- `tests/test_tools.py` — Private exceptions and untrusted tool/argument fields sanitized.
- `tests/test_tools_endpoint.py` — Sanitized unknown-tool response.

Existing security-relevant test functions were retained; the Redis SETEX test was replaced with its atomic Lua equivalent. Frozen replay fixtures and classifier/calibration assets were not regenerated.

### Validation

| Check | Result | Evidence and limits |
|---|---|---|
| Original negative controls | PASS | 20/20 failed before fixes, then passed; negative-before.log and negative-after.log. |
| Stdio leak negative control | PASS | Failed before correction; stdio-negative-before.log; passes in final affected suite. |
| Latest affected suite | PASS | 287 passed, 26 warnings, 22.10s after all source changes; final-security-and-errors.log. |
| Broad CI selection and coverage | PASS | 1,715 passed, 1 slow test deselected; 91.30% coverage against 65% threshold; acceptance-final.log. This preceded the final credential-normalization and stdio diagnostic edits, covered by the latest affected suite. |
| Full suite including slow replay | FAIL | 1,701 passed, 2 failed, 89.80% coverage; full-final.log. One obsolete single-event MCP assertion was corrected and rerun successfully. Frozen replay remains 50/300 mismatches. Full slow selection was not rerun after that assertion correction. |
| Mutation controls | PASS | Removing tool intent, MCP ownership or pinned connection makes the matching test fail; mutation-results.json and mutant logs. Only independent mutation copy changed. |
| Ruff core/api | PASS | Final current source tree. |
| Diff whitespace | PASS | Final git diff --check. |
| Mypy scoped gate | FAIL | 47 errors in 16 files, 42 modules; mypy-delivery.log. Baseline had 48 errors in 16 files. Normalized comparison found no new error messages; this does not make the gate pass. Ordinary invocation also encountered package duplication. |
| Windows local ASGI/process/TCP/TLS | PASS / VERIFIED | Actual endpoint, socket, TLS chain/hostname, stall, drip and process-reaping tests. TLS certificates were temporary test fixtures. |
| Cached model availability | PARTIALLY VERIFIED | English cached assets executed offline. German/multilingual model assets unavailable; canonical tier acceptance unverified. |
| Live Redis / multiple replicas | UNVERIFIED | Lua invocation/script contracts tested with mocks; no live Redis server acceptance. |
| Docker and Linux deployment | UNVERIFIED | Docker daemon unavailable; no Linux container/service validation. |
| Fresh remote CI / public external transport | UNVERIFIED | No push or publication; no remote run claimed. Transport tests use controlled local servers. |

Final type command: `python -m mypy core/ api/ --ignore-missing-imports --explicit-package-bases --follow-imports=skip`.
Final lint command: `python -m ruff check core/ api/`.
Tests ran in an isolated validation copy with local offline model assets, not against managed production.

### Remaining Phase 1 gaps

- Frozen replay has 50/300 mismatches, including expected HIGH becoming LOW. Unavailable canonical model assets complicate attribution; this is not established as purely environmental or pre-existing. Keep assertions unchanged and verify the canonical model tier before accepting the decision boundary.
- Mypy remains failing. Baseline comparison supports no added error messages, but the repository type gate is not green.
- Redis atomic behavior is implemented and unit-tested, but live Redis, multiple replicas and Linux deployment remain unverified.
- Cache reassessment and process-isolated fetching change cost/latency; existing benchmark claims were not remeasured or upgraded.
- Process timeout tests bound active work and verify reaping on Windows; process cleanup introduces small scheduler/reaping overhead. No cross-platform deadline guarantee is claimed from Windows tests alone.

### New security findings

- FIXED: stdio MCP internal errors echoed private request/exception content.
- FIXED: unknown tool names and argument keys could persist untrusted sensitive strings in audit/error details; generic reasons and hashed unknown audit identity now prevent that exposure.
- FIXED: own API keys ending in dash/underscore escaped the prior word-boundary detector.
- OPEN ACCEPTANCE FINDING: the fresh frozen replay result is 50 mismatches, including HIGH-to-LOW changes. Root cause and canonical tier correctness are unresolved; no exploit attribution is asserted.

### Before/after

Configured tenant corruption, absent output verdicts, permissive cached results, hidden audit failures and foreign review/MCP ownership can no longer silently authorize the tested paths. Tool execution now requires a persisted intent, and outbound connections preserve the validated address through the actual transport. These implementation and local test improvements do not establish canonical model, live Redis or production readiness.

### Evidence and backups

Original tracked snapshot, negative controls, successful/failed logs, final patch and changed-file checkpoint:
`D:\Projects_Bucket\Gatekeeper-phase1-backup-20261006`.

Independent runtime validation:
`D:\Projects_Bucket\Gatekeeper-phase1-validation-20261006`.

Independent mutation validation:
`D:\Projects_Bucket\Gatekeeper-phase1-mutations-20261006`.

No raw model or runtime evidence is added to this public documentation. TLS implementation references: [Python verified TLS contexts](https://docs.python.org/3/library/ssl.html#ssl.create_default_context) and [HTTP connection transport](https://docs.python.org/3/library/http.client.html#http.client.HTTPConnection).
