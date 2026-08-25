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
- [ ] Automated dependency-vulnerability scanning in CI — **not yet wired
      up** (`docs/OWASP_COMPLIANCE.md`'s LLM03 entry). Acceptable to ship
      without it only if this is a known, tracked gap, not silently absent.
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
- [ ] Multi-worker / distributed rate-limiting behavior — **known
      limitation, not yet built** (`docs/THREAT_MODEL.md` §8). If this
      release runs more than one worker process, rate limits and token
      quotas are per-process, not aggregate. Confirm this is acceptable
      for the target deployment before shipping multi-worker.

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
- [ ] Rollback plan — deploying a bad policy is covered
      (`core/policy_versioning.py`'s snapshot-before-overwrite, itself
      snapshotted before a rollback). Rolling back a bad **application**
      release (not a policy change) relies on standard container/image
      versioning at the deployment layer, outside this repo's scope —
      confirm the deployment target has this before release.

## Sign-off

| Role | What they're confirming | Status |
|---|---|---|
| Security reviewer | The Security section above, `docs/THREAT_MODEL.md`, `docs/OWASP_COMPLIANCE.md` | Pending human sign-off |
| QA / test owner | The Correctness section above, `docs/TRACEABILITY_MATRIX.md` | Pending human sign-off |
| Release manager | The Operational readiness section above, the known-limitations list is acceptable for this release's target deployment shape | Pending human sign-off |

This document does not self-certify a release as ready — it is the
checklist a human signs against. Every unchecked box above is either a
real, stated gap (fix or explicitly accept before shipping) or a decision
that depends on the specific deployment target (multi-worker, rollback
tooling) that only the release manager can make.
