# Gatekeeper V1.0 — Session Checkpoint

Signed off as **V1.0** at commit `c89465f`, tag `v1.0`. This document exists
so a fresh session (mine or a human's) can pick up this project without
re-deriving the last several weeks of work from scratch. It is a snapshot,
not a living doc — update it again at the next major milestone rather than
editing history into it piecemeal.

## What V1.0 actually is

An AI governance/security gateway sitting in front of LLM traffic: input
risk detection (fusion ensemble of 8 local transformer detectors + symbolic
rules + semantic judge), output guardrails (secrets/PII/toxicity/system-
prompt-leak), per-tenant policy and privacy config, an MCP tool gateway
(stdio + HTTP/SSE transports), Redis-backed distributed rate limiting and
token quotas for multi-replica deployments, and a full audit/review-queue
trail. Full narrative: `docs/ROADMAP_V2.md`. Architecture: `docs/
ENGINEERING_ASSESSMENT.md`, `Technical_Report.md`.

## Verification status (all re-runnable, none estimated)

| What | Command / evidence | Result |
|---|---|---|
| Full test suite | `python -m pytest tests/` | 1601 passed, 0 failed |
| Lint | `ruff check core/ api/` | clean |
| Coverage | CI-enforced floor | 91%+ |
| Live black-box pentest | `python scripts/live_pentest.py` | 46/46 |
| Redis live verification | `python -m scripts.live_redis_verification` | 12/12, plus real multi-process replica test |
| OpenAPI contract | `pytest tests/test_openapi_contract.py` | 4/4 |
| Perf SLO | `_evidence/perf_slo_benchmark.json` | zero 5xx across 4 endpoint load tests |
| Mutation testing | `docs/SDLC.md` Testing section | 8 mutants tested, 1 real gap found and closed |
| CI (GitHub Actions) | `.github/workflows/ci.yml`, `dependency-audit.yml` | green on `c89465f` |

Full go/no-go gate with every item individually cited: `docs/
RELEASE_CHECKLIST.md`.

## Sign-off status (as of this checkpoint)

| Role | Status |
|---|---|
| Security reviewer | Approved 2026-08-27 |
| QA / test owner | Approved 2026-08-27 |
| Release manager | Approved 2026-08-27 |

All three sign-offs were approved on 2026-08-27 against the cited,
re-runnable evidence and the deployment runbook. V1.0 is fully signed off.

## Known, accepted limitations (not gaps — see `docs/THREAT_MODEL.md` §8 for full detail)

- Circuit breakers are process-local even when Redis is configured — each
  replica discovers a judge-backend outage independently.
- No cross-tenant embedding partitioning in the semantic cache/vector
  stores — a fuzzy-match hit can influence another tenant's cached verdict
  by similarity (never raw text).
- Rate limiter's local-fallback mode is LRU-capped (`RATE_LIMIT_MAX_TRACKED`).
- MCP stdio transport has no per-caller authorization (deliberate scope
  boundary for that transport only — MCP HTTP/SSE does have it).
- A full 20MB audit-log scan for a truly-zero-match query is still real
  work under concurrency (an index would remove this; not built).
- Hallucination/grounding check is disabled by default
  (`HALLUCINATION_CHECK_ENABLED=False`) pending a correct output-domain
  reference corpus — the one that exists borrows the wrong (input-side) one.

## Explicitly deferred (not started, not scheduled — 280–400h tier)

RBAC + API key management UI, hierarchical tenant/org management, advanced
cost controls/spend forecasting, dynamic latency-based model routing,
distributed tracing (OpenTelemetry), attack-campaign/IP clustering
detection, compliance reporting export (SOC2/ISO27001), automated red-team
evaluation harness, proper frontend/backend product separation.

## Repository state at this checkpoint

- Branch: `main` only. The four stale backup/worktree branches from the
  pre-trailer-rewrite era were deleted this session (verified safe: every
  commit message matched `main`, and two independent file-diff spot-checks
  showed the ONLY difference was the `Co-Authored-By: Claude` trailer that
  a prior rewrite deliberately stripped — this project's standing rule is
  no AI-authorship trailer in any commit, ever).
- Untracked and deliberately left alone: `review_queue.json` (live
  application state — 93 real PENDING review records, not scratch),
  `.claude/` (local Claude Code config, not project source).
- Two old tags remain from before the backup-branch cleanup:
  `backup-phase1-branch-pre-rewrite-tag`, `backup-pre-trailer-rewrite-tag`.
  Harmless (tags don't keep a branch "alive" the way a ref does), left
  untouched — not part of this cleanup's scope.

## If you are a fresh session picking this up

1. Read this file first, then `docs/ROADMAP_V2.md` for the full narrative
   and `docs/RELEASE_CHECKLIST.md` for the current go/no-go gate.
2. Re-run the verification table above before trusting any of it —
   "re-verify this exact count before release; it will have grown" is the
   standing instruction in `RELEASE_CHECKLIST.md` itself.
3. This project's non-negotiable rule, enforced every commit: never
   include `Co-Authored-By: Claude`/`Anthropic` or any AI-authorship
   trailer in a commit message. Verify with
   `git log -1 --format=%B | grep -ci "claude\|anthropic"` → must be `0`.
4. Full lint + full test suite must pass before any commit
   (`ruff check core/ api/` && `python -m pytest tests/`).
5. This session's working discipline throughout: verify claims against
   real running infrastructure (real Docker Redis, real separate OS
   processes, real HTTP requests) wherever the claim is about production
   behavior — not mocks or theoretical reasoning alone. Continue that.
