# GATEKEEPER PHASE 1.1 COMPLETE

Date: 2026-10-06. Scope: local correctness and release-blocker resolution.
HEAD remains `6b1cf091c6717b494d9e6bce123e941d379419d2`.
No commit, push, publication, history change or GitHub setting change occurred.

## Replay

| Result | Before | After |
|---|---:|---:|
| Risk mismatches / 300 | 50 | 0 |
| HIGH -> LOW | 34 | 0 |
| LOW -> HIGH | 16 | 0 |

The standalone final replay passed all 300 risk levels and all 300 decision
sources. The full-suite slow gate independently ran a fresh model process.
Expected fixtures, model weights, fusion thresholds and policies were not
regenerated or relaxed.

### Every original mismatch

Private evidence root: `D:\Projects_Bucket\Gatekeeper-phase11-evidence-20261006`.
`all-50-mismatch-table-final.csv` and
`all-50-mismatch-classification-final.json` contain all 50 case IDs, languages,
attack classes, expected/before/after risk and source, feature scores, cache
state, tenant/capability context, configuration fingerprint, judge state,
model tiers, root cause and resolution. All 50 are resolved.

The 34 HIGH -> LOW cases comprise 17 benign-labelled rows, nine harmful-content
rows and eight injection rows. HIGH is the frozen governance decision, not an
assertion that every such row is malicious: 25 were fail-closed after unavailable
judging, nine were fusion critical. These exact decision sources are restored.

### Root causes and controls

1. **47 original mismatches: detector availability / tier drift.** Missing
   German detectors and multilingual-head assets selected `six_feature` instead
   of the richer `nine_feature` profile. All 34 HIGH -> LOW cases belong here.
   The richer feature set restores the original critical/judge-zone paths;
   judges remain unavailable and ambiguous results still fail closed to HIGH.
   Required pinned assets were restored and the canonical gate now rejects
   missing models or a degraded tier, including a tier change during replay.
2. **Three original mismatches: Prompt Guard 2 tokenizer drift.** Identical
   pinned weights under Transformers 4.57.6 removed whitespace boundaries and
   gave scores approximately 0.99784, 0.76701 and 0.96442 for case IDs
   `205a101942bbfda6`, `c7233408ab9757af`, `d82628dddd10b50d`. Transformers
   5.2.0 preserves boundaries and reproduces retained historical signal scores
   approximately 0.01292, 0.40837 and 0.01634. Repeated, single, mixed-length
   and padding checks ruled out random/batch-padding explanations.
3. **Two intermediate mismatches: uncased toxicity preprocessing.** Switching
   runtime alone changed Bert's default lowercasing to false. Two German benign
   rows (`34826d0eb0eefce4`, `8b308991a74c43c7`) entered the judge zone and
   failed closed HIGH. Explicit `do_lower_case=True`, grounded in the pinned
   tokenizer's `bert-base-uncased` declaration, restores scores exactly to
   0.2923046052455902 and 0.037907686084508896. This fixes model preprocessing,
   not case IDs or decision thresholds.

Intermediate rich-tier runs were retained: v4 had 3/300 mismatches; uncorrected
v5 had 2/300; both had zero HIGH -> LOW. Generic-tokenizer substitution under
v4 did not resolve Prompt Guard drift and was not implemented.

Replay caches were bypassed in both diagnostic and final runs, so none of the
50 differences depends on cache state. Phase 1 cache ordering, tenant scope,
raw-input identity and symbolic vetoes were preserved. Policy anchors match the
historical class grouping. A later committed fusion recalibration predates this
task; its artifact is unchanged and does not alone explain these differences.

The degraded diagnostic captured all 300 decisions and the exact 50 mismatches,
then failed its final manifest step because the initial source-only snapshot
omitted the ignored evaluation dataset. That process exit is not called a pass;
its completed row capture was independently counted. Final strict runs validate
the matching dataset before inference.

### Final canonical configuration

The complete machine-readable specification is `config/canonical_replay.json`;
usage and measurement boundaries are in `docs/CANONICAL_REPLAY.md`.

- Nine-feature per-class fusion: anchor feature plus all eight pinned model
  features, no optional-detector degradation. Existing manifest weight SHA256s
  are verified before scoring.
- Anchor encoder: all-mpnet-base-v2 at
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`; multilingual encoder:
  paraphrase-multilingual-MiniLM-L12-v2 at
  `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
- Torch 2.6.0+cpu, Transformers 5.2.0, sentence-transformers 5.2.0,
  huggingface-hub 1.33.0, NumPy 2.4.4, tokenizers 0.22.2,
  safetensors 0.8.0. Critical package versions are checked, not inferred.
- Seed 42, two Torch threads, deterministic algorithms, parallel fusion,
  domain guardrail off; both judges explicitly unavailable; cache bypassed.
- Nine-feature HIGH / MEDIUM thresholds: harmful 0.58 /
  0.2508880281868449; jailbreak 0.13 / 0.017693995702321372;
  injection 0.48 / 0.18544612817653217. No threshold was edited.
- Fusion v3, multilingual head, detector manifest, five policy JSONs, frozen
  oracle from commit `5984269`, and ignored `data/eval_suite.jsonl` are pinned
  by normalized SHA256 in the profile.
- Final standalone security/configuration fingerprint:
  `5a4143c68993297db52dda0a3bcea0a735610668de3231c43b5a15c44632e749`.
  This includes source, settings, policies, artifacts, availability and runtime
  versions; tenant/raw identities are additional cache-scope components.

The shared profile/preflight defines future comparable replay, benchmark,
calibration and published decision claims. Historical benchmark/calibration
artifacts were not rerun or upgraded. Existing exploratory tools are not evidence
of canonical-equivalent accuracy. The contradicted historical v4/v5 equivalence
claim is marked stale in `docs/MAINTENANCE.md`; headline numbers are unchanged.

## Type checking

- Reported before: 47 errors, compared with a pre-Phase 1 baseline of 48.
- After: **0 errors in 42 source modules**. Newly introduced errors remaining:
  **0**. Pre-existing errors remaining in this gate: **0**.
- All 47 reported diagnostics matched pre-existing baseline messages. Concrete
  nullable contracts, guarded narrowing and container/interface types resolved
  them. Following installed imports additionally exposed five pre-existing
  boundary errors in spaCy nullability, Requests JSON payloads and the API's
  fallback status; all five were fixed.
- `type-error-classification-final.json` classifies all 52 encountered
  diagnostics. No blanket Any, new ignore or disabled check was added. The final
  invocation follows imports; the earlier `--follow-imports=skip` workaround is
  absent. `--ignore-missing-imports` accommodates missing third-party stubs, and
  mypy's existing default still does not check wholly unannotated bodies. This
  is a clean scoped gate, not a claim that every function is fully typed.

## Security invariants

| Phase 1 invariant | Status | Final evidence / boundary |
|---|---|---|
| Failed tenant-store load cannot unsuspend/admit | PASS | Missing, corrupt, malformed and concurrent reload tests retain denial. |
| Guarded toxicity allows only explicit SAFE | PASS | Absent/malformed/error/timeout/offline verdicts block. |
| Permissive cache cannot bypass current veto | PASS | Symbolic check precedes cache; exact/fuzzy LOW is reassessed. |
| HIGH monotonic within assessment version | PASS | Local concurrent writes and Redis script contracts; changed source/policy/model/runtime creates a new version. |
| Authoritative audit failure observable; side effects require intent | PASS | Storage-fault tests, API propagation and handler-order mutation. |
| Review ownership | PASS | Foreign tenant review probes are opaque. |
| Authenticated, tenant-bound, bounded MCP sessions | PASS | Authentication, owner, bounds and dispatch tests; ownership mutation. |
| SSRF destination bound to actual connection/deadline | PASS | Validated address pinning, local TCP/TLS, DNS stall, slow drip and child reaping; connection mutation. |
| Live Redis atomicity across replicas | UNVERIFIED | Lua contract tests are local; no running Redis/multi-process acceptance. |
| Linux transport/deployment behavior | UNVERIFIED | Windows runtime evidence only; Docker engine unavailable. |

A second code review checked symbolic-before-cache, HIGH-only backend reuse,
required tool-intent persistence, MCP owner-before-dispatch, explicit SAFE output
parsing and pinned outbound connection ordering. Type fixes preserve fusion's
existing fallback feature order. No security control was traded for replay.

Cache regressions exercise cold, warm exact, forced fuzzy lookup, cross-tenant,
raw/redaction collisions, policy/threshold/model/runtime version changes,
LOW -> HIGH, retained HIGH after lower reassessment and current hard vetoes.
Concurrent HIGH persistence is covered by the Phase 1 suite. Runtime version
reuse failed with the original cache implementation and passes after correction.
The lowercase contract likewise failed before its loader fix.

## Validation

| Check | Exact result | Evidence |
|---|---|---|
| Affected security suite | PASS: 578 tests; 26 warnings; 81.15s | final-security.log / final-security.xml |
| Full suite, including slow replay | PASS: 1,747 tests; 32 warnings; 503.79s | full-final.log / full-final.xml |
| Canonical replay | PASS: 300/300 risks and sources, twice; 0 HIGH -> LOW | canonical-final/ and canonical-full-suite/ |
| Mutation / negative controls | PASS: 8/8 mutations killed; runtime/casing regressions fail before fix | mutation-results.json and mutant logs |
| Ruff / whitespace | PASS: core/api, new scripts/tests, git diff --check | ruff-final.log / ruff-new-files-final.log / whitespace-final.log |
| Type checking | PASS: 0 errors in 42 modules, imports followed, canonical runtime | mypy-final-canonical-runtime.log |
| Coverage | PASS: 91.37%, against 65% required | coverage-final.json |
| MCP | PASS: 72 tests | final-security.xml; file membership in security-selection.json |
| cache | PASS: 73 tests | final-security.xml; file membership in security-selection.json |
| tenancy | PASS: 49 tests | final-security.xml; file membership in security-selection.json |
| audit | PASS: 33 tests | final-security.xml; file membership in security-selection.json |
| SSRF | PASS: 47 tests | final-security.xml; file membership in security-selection.json |
| output guardrails | PASS: 35 tests | final-security.xml; file membership in security-selection.json |

Commands (run from the isolated validation copy except static checks):

```text
python -m scripts.canonical_replay --output <private-evidence>/canonical-final
python -m pytest -o addopts='' --cov=core --cov=api --cov-report=term-missing --cov-report=json:<private-evidence>/coverage-final.json --cov-fail-under=65 --junitxml=<private-evidence>/full-final.xml
python <private-evidence>/run_security.py
python <private-evidence>/run_final_mutations.py
python -m ruff check core/ api/
python -m ruff check --isolated --select E,F --line-length 136 scripts/canonical_replay.py tests/test_canonical_replay_profile.py tests/test_phase11_cache_regressions.py tests/test_detectors.py
python -m mypy core/ api/ --ignore-missing-imports --explicit-package-bases
git diff --check
```

Model-dependent checks and final suites use the pinned v5 environment; the
mutation copy uses controlled/mock detectors. All eight mutations were killed
by assertion failures, not import/syntax failures: tool intent, MCP owner,
connection pinning, required model availability, artifact drift, runtime cache
version, uncased tokenizer contract and mid-replay tier drift.

`security-selection.json` records the exact files used for the affected suite.
Category counts use its JUnit file membership; groups overlap and should not be
summed. `mutation-results.json` and individual mutant logs retain negative proof.
Raw logs, signals, model caches and runtime outputs stay outside public Git.
An overlapping security run was deliberately interrupted under RAM pressure;
`security-interrupted-memory.log` is retained and is not a validation result.

## Infrastructure and evidence

- **VERIFIED:** final local Windows unit/API boundary, process, TCP and TLS
  tests; real pinned CPU model inference, complete nine-feature tier and frozen
  300-case replay. Source comparison found no drift between the user tree and
  the validation copy for core/api/scripts/config/tests.
- **PARTIALLY VERIFIED:** inference environment reproducibility. All critical
  versions/revisions/hashes are checked, but the diagnostic venv inherits system
  dependencies and an unused optional Optimum package has an incompatible v5
  requirement. This is Torch-only proof; a fresh clean full deployment install
  and ONNX equivalence are not claimed.
- **UNVERIFIED:** live Redis, multiple replicas, Linux/Docker deployment,
  public external transport and fresh remote CI. None of the 50 mismatches
  required these systems. Distributed Redis monotonicity still requires live
  server/concurrent-client acceptance before making that operational guarantee.

Original Phase 1 source snapshot: `<private-evidence>/before/` (unchanged).
Final Phase 1.1 diff/checkpoint, classifications and successful/failed logs are
in the private evidence root. Model/calibration artifacts and frozen fixtures
have no diff. No repository files were staged.

## Remaining blockers

None for the scoped Phase 1.1 local correctness gate. Live Redis/replicas, Linux
deployment, fresh clean deployment installation and renewed evaluation claims
remain unverified work, with the operational dependencies identified above.
They are not converted into local test passes.

## Release decision

RELEASE CANDIDATE

This decision applies to the verified local pinned CPU profile and preserved
security boundaries. It does not establish production operation, distributed
Redis acceptance, Linux deployment or generalization accuracy.
