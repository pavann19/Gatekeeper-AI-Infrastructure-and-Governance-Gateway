# Software Development Lifecycle

This document maps Gatekeeper's actual, already-happened development
history onto a named SDLC model, with evidence citations — not a process
description written to look good, but a retroactive account of what this
project's 121+ commits, 8-phase roadmap, and this session's independent
hardening pass actually did, and where that falls short of a textbook
process.

## Model: Iterative / Agile-with-gates

Gatekeeper was built as a sequence of scoped phases (`docs/ROADMAP_V2.md`),
each shipping a vertically complete slice (engine → output security → LLM
Gateway → Tool Gateway → Policy-as-Code → Human Review → dual UI →
continuous hardening), with two gates enforced on every change rather than
only at release boundaries:

1. **A CI gate per commit** (`.github/workflows/ci.yml`): lint (`ruff check
   core/ api/`) and the full test suite with a coverage floor
   (`--cov-fail-under=65`, currently at 91%+) must pass before any commit
   lands on `main`.
2. **An evidence gate per finding**: every fix in `docs/ROADMAP_V2.md`'s
   Phase 8 section is written as a before/after with either a real
   benchmark number, a real reproduction, or a real test — not asserted
   fixed on inspection alone.

This is not Waterfall (requirements were never frozen up front — the
8-phase scope itself was reprioritized mid-project by leverage/risk, see
the "Gatekeeper 2.0 roadmap decision" that reordered LLM/Tool Gateway
behind cheaper, lower-risk work) and it is not textbook Scrum (no fixed
sprint cadence or ceremony) — "iterative, gated, evidence-driven" is the
honest description.

## Lifecycle stages, as they actually happened

### 1. Requirements & threat modeling
- Origin: an external 8-phase scoping proposal (160h MVP / 280–400h full
  scope) was evaluated against this project's own architecture and
  hardware constraints, not adopted verbatim — phase ORDER was
  reprioritized by leverage-per-hour and hardware risk (LLM/Tool Gateway
  moved behind cheaper, already-scoped work).
- Formalized retroactively this session as `docs/THREAT_MODEL.md`: trust
  boundaries, capability tiers, a STRIDE pass per subsystem, and a
  findings register — written from reading the actual code, not from the
  original design intent.

### 2. Design
- Every non-trivial module's docstring states its own design rationale
  inline (e.g. `core/risk.py`'s staged pipeline docstring, `core/auth.py`'s
  "THE VULNERABILITY THIS REPLACES" header) — design decisions are
  co-located with the code they justify rather than living only in a
  separate design doc that can drift.
- `docs/ENGINEERING_ASSESSMENT.md` documents real design corrections found
  during development (e.g. a tautological threat-present check that
  silently defeated a calibration effort — §1j).

### 3. Implementation
- Small, scoped commits (121+ on `main`) each doing one thing, with commit
  messages stating the *why*, not just the *what* — visible in `git log`.
- No speculative abstraction: the codebase repeatedly favors "three similar
  lines" over a premature shared helper until a third real use case
  justifies one.

### 4. Testing
- **Unit + integration**: 1522 tests as of this session (up from 845 at
  the start of this session's work, up from ~710 before Phase 7).
  Coverage-gap-driven expansion targeted modules with zero or thin
  dedicated coverage rather than padding existing files.
- **Contract**: `tests/test_openapi_contract.py` validates the generated
  OpenAPI schema is spec-conformant and that documented size bounds match
  real enforced validation.
- **Mutation testing**: 8 real mutants manually introduced into
  security-critical functions (path-traversal guard, SSRF allowlist,
  rate-limit boundary, secret-detection block, auth credential check, the
  HIGH-risk cache never-downgrade guard) this session — 6 killed cleanly
  by existing tests, 1 revealed a genuine gap (closed with 3 new tests,
  `tests/test_cache_locked_high_never_downgrades.py`), 1 target was
  correctly refused by the safety classifier given its blast radius and
  left to existing coverage. `mutmut` has no native Windows support; a WSL
  environment rebuild for the project's full ML dependency stack was
  judged not worth the detour versus manual, targeted mutation testing on
  the highest-value functions.
- **Adversarial / black-box**: `scripts/live_pentest.py`, run against a
  real running instance with real provisioned API keys and tenants —
  auth bypass, a 10-prompt real jailbreak corpus, output-guardrail probes,
  SSRF, path traversal, oversized payloads, rate limiting, tenant
  suspension. 46/46 passing after two real bugs it found were fixed.
- **Load / performance**: `scripts/load_test.py` against a real running
  instance at real concurrency — see `docs/OWASP_COMPLIANCE.md`'s
  Performance SLO table. This exact benchmark found a real KeyStore
  concurrency bug (4% spurious 401s under load) that no unit test had
  caught, fixed and re-verified live at 0/300.
- **UI / E2E**: real-browser walkthroughs (Claude Browser tooling) of
  every page across all three capability tiers against a real running
  instance — found the `http.get` real-tool registration gap (built,
  tested, never wired into any startup path).

### 5. Hardening (continuous, not a final phase)
Phase 8 in `docs/ROADMAP_V2.md` is explicitly "continuous, not a final
phase" — every finding above was fixed with a regression test in the same
change, not queued into a backlog. The findings register in
`docs/THREAT_MODEL.md` §7 and this session's additions to it are the
audit trail.

### 6. Release / deployment
- Docker image runs as a dedicated non-root user, verified against real
  builds and real named volumes (catching a real second bug — a volume
  mount defaulting to root ownership — in the process, not just reviewing
  the Dockerfile).
- Every "should this default on" flag (`REGISTER_DEMO_TOOLS`,
  `REGISTER_REAL_TOOLS`, `DOMAIN_GUARDRAIL_MODE`,
  `HALLUCINATION_CHECK_ENABLED`) defaults to the conservative setting.
- See `docs/RELEASE_CHECKLIST.md` for the concrete go/no-go gate this
  project uses before a release is called ready.

## Where this falls short of a textbook process (stated honestly)

- **No formal requirements sign-off step** — the roadmap was a living
  document reprioritized mid-flight based on evidence (hardware limits
  discovered under load, work already done being double-counted in the
  original estimate), not a frozen baseline. This was the right call for
  a project of this size and risk profile, but it means there is no
  single artifact a stakeholder "signed" before work began.
- **No automated dependency-vulnerability scanning** in CI today
  (Dependabot/pip-audit) — `docs/OWASP_COMPLIANCE.md`'s LLM03 (Supply
  Chain) entry states this as an accepted gap, not a hidden one.
- **`mutmut` mutation testing could not run natively** on this
  development machine (Windows, no native support) — manual mutation
  testing substituted for it this session, which is real evidence but not
  as exhaustive as an automated mutation-testing pass across the whole
  codebase would be.
- **Process-local rate limiting/quotas** — a real, stated architectural
  limitation (not an SDLC gap) that affects how the perf SLO numbers in
  `docs/OWASP_COMPLIANCE.md` extrapolate to a multi-worker deployment.

See `docs/TRACEABILITY_MATRIX.md` for the requirement/threat → code → test
mapping this lifecycle produced, and `docs/RELEASE_CHECKLIST.md` for the
concrete sign-off gate before any release.
