# Release / QA Sign-Off Checklist

A concrete go/no-go gate, filled in against this project's actual current
state (not a template). Re-run this checklist before every release, not
just once — most items point at a command or file that can be re-verified
in under a minute.

## Security

- [x] Every endpoint's auth/capability/tenant-suspension enforcement is
      tested — `pytest tests/test_endpoint_auth_matrix.py` (81 tests).
- [x] No known unpatched path traversal, SSRF, or injection vulnerability
      — see `docs/THREAT_MODEL.md` §7's findings register: every entry is
      `Fixed`, none open.
- [x] Live black-box pentest passes — `python scripts/live_pentest.py`
      against a running instance → 46/46 (`_evidence/live_pentest_results.json`).
- [x] Secrets are never logged or persisted in plaintext — API keys are
      SHA-256 hashed at rest (`core/auth.py::hash_key`), audit records
      carry only `prompt_hash`/`arguments_hash`, never raw content
      (`tests/test_logger.py`).
- [x] Every request schema rejects unrecognized fields (`extra="forbid"`)
      — closes the exact privilege-escalation vector (`{"role": "INTERNAL"}`)
      this project's original auth vulnerability was.
- [x] Automated dependency-vulnerability scanning in CI —
      `.github/workflows/dependency-audit.yml` runs `pip-audit` against
      `requirements.txt`, `requirements-api.txt`, and
      `requirements-ci.txt` on push/PR to `main` and weekly on Mondays.
- [x] OWASP API Security Top 10 and OWASP LLM Top 10 mapped with cited
      evidence — `docs/OWASP_COMPLIANCE.md`.
- [x] The new MCP HTTP/SSE transport (`core/mcp_http_server.py`) received
      the same STRIDE-level review as every other endpoint before merge,
      not a pass on trust — `docs/THREAT_MODEL.md` §6.2. Found and fixed
      3 real issues in review: the SSE session-relay queue was dead code
      (any `sessionId` was accepted, responses never actually reached the
      SSE stream), it shared a rate-limit bucket with the main API
      (a real cross-traffic-class starvation risk once Redis-backed), and
      its parse/size error responses weren't valid JSON-RPC. All 3
      verified fixed live against a real running server, not just by
      re-reading the diff.
- [x] The output toxicity judge does not silently fail open when its
      primary backend is unreachable — `output_judge()` returning
      `JUDGE_OFFLINE` used to fall through `assess_output`'s `if verdict
      == "DANGEROUS"` check to ALLOW. Fixed with a local fallback judge
      (`core/semantic_judge.py::_fallback_output_judge`, reusing the
      already-warmed `toxic_bert` detector — no cold-start cost at the
      moment it's needed) engaged before the offline sentinel is ever
      returned. `docs/THREAT_MODEL.md` §7 finding #16, `docs/
      TRACEABILITY_MATRIX.md`, 7 new tests, verified live against the
      real model (71ms warm-call latency).

## Correctness

- [x] Full test suite passes — `python -m pytest tests/` → 1601 passed,
      0 failed (re-verify this exact count before release; it will have
      grown).
- [x] Lint clean on the CI-gated scope — `ruff check core/ api/` → All
      checks passed.
- [x] Coverage floor met — CI enforces `--cov-fail-under=65`; actual
      coverage is 91%+ as of the last CI run.
- [x] OpenAPI contract is spec-valid and matches real enforced validation
      — `pytest tests/test_openapi_contract.py` (4 tests).
- [x] Mutation testing performed on security-critical logic, not just
      line coverage — 8 real mutants tested this session (`docs/SDLC.md`'s
      Testing section); the one real gap found (`cache_locked_high` never
      being asserted) is now closed.

## Performance

- [x] Real load test run against a live instance, not estimated —
      `_evidence/perf_slo_benchmark.json`. Zero 5xx / connection errors
      across all 4 endpoint runs.
- [x] Timeouts fail closed (503, never a fabricated verdict) — verified
      both in unit tests (`tests/test_request_limits.py`) and live
      (the p95/p99 assess latency in the perf table is real cold-path
      time, under the 30s deadline).
- [x] Multi-worker / distributed rate-limiting behavior — release
      decision documented in `docs/DEPLOYMENT_RUNBOOK.md`: this release is
      approved for one API worker process per deployed instance without
      Redis configured. `REDIS_URL` now enables a distributed rate
      limiter and token quota tracker (`core/rate_limit.py::
      RedisRateLimiter`, `core/token_quota.py::RedisTokenQuotaTracker`)
      for genuine multi-worker/multi-replica deployments.
- [x] Redis-backed rate limiting/quotas verified against a real live
      Redis server (`redis:7-alpine`, Docker) under real concurrent
      access — `scripts/live_redis_verification.py`,
      `_evidence/live_redis_verification_results.json`, 12/12 passed:
      real connection-pool sharing across all 4 subsystems, real
      token-bucket burst/deny/refill math with a real wall-clock wait,
      200 real concurrent threads against a 50-token bucket never
      overspent capacity (the Lua script's atomicity genuinely holds
      server-side, not just in a Python simulation), 100 concurrent
      token-quota increments summed to exactly the correct total (no
      lost updates), and a real Redis-becomes-unreachable-mid-process
      case fell back to the local limiter instead of crashing. Then
      verified the actual multi-replica scenario directly: two
      independent `uvicorn` processes plus a separate MCP HTTP server
      process, all pointed at the same real Redis — a rate-limit burst
      against replica 1 was immediately visible as exhausted on replica
      2 (a genuinely different OS process), with the correct
      `Retry-After` header, while the MCP server's own rate limiter
      (same API key, same Redis, same instant) was completely
      unaffected, confirming the `mcp_rate_limiter`/`assess_rate_limiter`
      namespace isolation fix holds under real infrastructure, not just
      unit-test object-identity assertions.

## Operational readiness

- [x] Docker image runs as non-root, verified against a real build and
      real named volumes (not just a Dockerfile review) —
      `docs/ROADMAP_V2.md`'s Phase 8 entry.
- [x] Every "should this be on by default" flag defaults to the
      conservative setting — `REGISTER_DEMO_TOOLS`, `REGISTER_REAL_TOOLS`,
      `DOMAIN_GUARDRAIL_MODE`, `HALLUCINATION_CHECK_ENABLED` all default
      off/disabled.
- [x] `/health` reports real per-dependency status (policy files, spaCy,
      embedding model, semantic judge) without itself becoming a load
      source under concurrent monitoring (reads circuit-breaker state
      directly, never calls the side-effecting `is_open()`).
- [x] CI gate blocks merge on lint or test failure —
      `.github/workflows/ci.yml`, triggers on push/PR to `main`.
- [x] Rollback plan — policy rollback is covered by
      `core/policy_versioning.py`; application rollback is documented in
      `docs/DEPLOYMENT_RUNBOOK.md` and requires immutable container image
      tags plus redeploying the previous known-good tag.

## Sign-off

| Role | What they're confirming | Status |
|---|---|---|
| Security reviewer | The Security section above, `docs/THREAT_MODEL.md`, `docs/OWASP_COMPLIANCE.md` | Pending human sign-off |
| QA / test owner | The Correctness section above, `docs/TRACEABILITY_MATRIX.md` | Pending human sign-off |
| Release manager | The Operational readiness section above, the known-limitations list is acceptable for this release's target deployment shape | Pending human sign-off |

This document does not self-certify a release as ready — it is the
checklist a human signs against. The remaining Pending statuses in the
sign-off table require the named humans to confirm the cited evidence and
deployment runbook before release.
