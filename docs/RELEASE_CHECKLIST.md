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

## Correctness

- [x] Full test suite passes — `python -m pytest tests/` → 1522 passed,
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
      approved for one API worker process per deployed instance. More than
      one worker/replica requires the Redis-backed distributed limiter and
      quota accounting first (`docs/THREAT_MODEL.md` §8).

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
