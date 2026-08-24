# Gatekeeper 2.0 Roadmap

Source: an external ("Gemini") scoping proposal, reviewed 2026-08-14 and
reordered by leverage-per-hour against this project's actual constraints
(solo build, 12GB RAM dev machine that has already forced two unexpected
shutdowns under heavy compute — see `docs/ENGINEERING_ASSESSMENT.md` §1v's
"unrelated finding"). Kept because it maps closely to real commercial AI
security gateway scope (Lakera / Robust Intelligence / Prompt Security
shaped), not because the hour estimates are trusted as-is — treat every
estimate below as a floor, not a target, and re-benchmark after each phase
the way §1v did rather than committing to the full scope blind.

Overlap note: Auth, rate limiting, audit logging, Prometheus/Grafana,
multi-tenancy, and PII redaction already exist and are NOT re-counted here
even where the original proposal's phase 1/8 implied rebuilding them.

---

## Phase 1 — Strengthen the existing security engine (7/8 done — one item blocked on data, not effort)
Original estimate: ~20–30h

- [x] Per-class risk vector groundwork identified (`class_scores` already
      computed in `core/fusion.py`, not yet surfaced)
- [x] Surface per-class risk vector in `details` / API response — done
      2026-08-14, see `docs/ENGINEERING_ASSESSMENT.md` §1w
- [x] Multilingual encoder — investigated 2026-08-14, see
      `docs/ENGINEERING_ASSESSMENT.md` §1x and
      `scripts/analyze_multilingual_fusion.py`. Findings: (a) the deployed
      4-feature fusion ensemble already narrows the German gap
      substantially vs. anchors-only (German AUC 0.632 -> 0.819
      out-of-fold, previously unmeasured for the ensemble) purely because
      `protectai_injection` is already in it — no encoder swap needed,
      confirming §1b/§1d's superseded recommendation; (b) adding Prompt
      Guard 2 as a 5th feature was tested live (gated access confirmed
      working) and does NOT move German performance (0.819 -> 0.819,
      confidence intervals identical) and the pooled AUC lift isn't
      statistically decisive either — NOT wired into `core/fusion.py`,
      negative result recorded rather than shipped. Encoder swap item is
      closed; a real, decisive residual gap remains (0.950 vs 0.819 AUC,
      84.7% vs 47.4% recall@5%FPR, English vs German) — replaced below
      with the honestly-scoped follow-up.
- [x] German-specific detection gap (renamed from "multilingual encoder"
      per the above) — SHIPPED 2026-08-24 on
      `phase1-german-gap-and-experiments` (not yet merged to `main`), see
      `docs/ENGINEERING_ASSESSMENT.md` §1aa (investigation) and §1ab
      (the fix and what shipped). Data blocker closed first (234→4,221
      German rows, 3 new sources). Reweighting the existing 4 features on
      German rows alone gave zero gain (AUC 0.670 vs 0.671) — decisive
      proof the features, not the weights, were the constraint, which
      reframed every subsequent attempt around adding information rather
      than redistributing it. Splitting German by task
      (`scripts.analyze_german_by_task`) found the pooled "German AUC"
      figure was 69% a different problem (`germeval18` offensive-content
      rows, not prompt injection) — corrected, and now tracked as two
      separate items. For German PROMPT INJECTION, the actual item:
      recall@5%FPR 49.6%→94.0% (AUC 0.813→0.987), using 2 more off-the-
      shelf detectors (`deepset_injection`, and a revived
      `prompt_guard_2` — §1x's 2026-08-14 rejection of it was right about
      German but wrong to treat that as the deciding criterion; it is the
      largest English/pooled gain available). Shipping required fixing
      `core/fusion.py`'s all-or-nothing fallback first: it now supports
      **upgrade tiers** — richer optional feature sets tried best-first,
      degrading one step (not to anchors-only) when an optional detector
      (e.g. a `prompt_guard_2` licence not yet accepted) is unavailable.
      Verified live on real detectors, not just mocks. German OFFENSIVE
      CONTENT (0.584→0.742 AUC) remains a distinct, open, weaker item —
      see §1ab's "what this does not close" — as does the purpose-built
      multilingual feature that beats this result but isn't shipped yet
      (stacking caveat, needs its own tier).
- [x] Clean threat taxonomy — done 2026-08-14, see
      `docs/ENGINEERING_ASSESSMENT.md` §1y. Two fixes: (1) added a
      `jailbreak` anchor class to `policies.json` — anchor layer previously
      modeled only harmful_content/prompt_injection despite jailbreak
      being 36% of attacks; measured out-of-fold before keeping (no
      regression, jailbreak recall@5%FPR 74.7%→80.6%), fusion policy
      retrained to match; (2) split `symbolic_rules.json`'s
      `jailbreak_patterns` (previously a mix of genuine jailbreak and
      instruction-override regexes reported under one misleading detail
      string) into `jailbreak_patterns` + `instruction_override_patterns`.
      8 new tests, 373 passed overall (unchanged).
- [x] Separate `risk` from `topicality` — done 2026-08-15, see
      `docs/ENGINEERING_ASSESSMENT.md` §2a. Verification finding: already
      correctly implemented, with a named regression test since before
      this roadmap existed (`test_off_topic_benign_prompt_is_not_a_safety_
      risk`, "THE core regression test"). No code change needed. Audited
      every consumer (policy, metrics, cache, API) — topicality only ever
      influences risk_level in the one deliberate, documented, opt-in
      `DOMAIN_GUARDRAIL_MODE=enforcing` case. Added 3 tests closing the one
      real gap: separation was tested at the decision level but not the
      enforcement layer it flows through afterward. 391 passed (up from
      388).
- [x] Investigate/remove dead dynamic threat feed — done 2026-08-14, see
      `docs/ENGINEERING_ASSESSMENT.md` §1z. Removed entirely (`core/
      updates.py`, `/api/v1/update`, all `dynamic_threat_score` refs — it
      was never auto-populated and never consumed by any decision even
      when populated). Investigation surfaced a real bug along the way:
      `is_educational` was wired to a different, permanently-empty dead
      function instead of the correct, already-implemented
      `check_educational_context` — silently disabling the entire
      educational-safe-harbor MEDIUM path. Fixed (one line). Measured
      impact without needing a live judge: only 4/901 ambiguous-zone rows
      in the eval suite flip, all 4 benign, zero attacks affected, no HIGH
      decision reachable by the fix at all. 7 new tests, 386 passed
      (up from 381). A second, adjacent dead-anchor-list finding
      (`policies.json`'s unused `safe_anchors`) was noted but NOT acted
      on — flagged for a future dedicated pass, not folded in here.
- [x] Recalibrate thresholds — done 2026-08-16, see
      `docs/ENGINEERING_ASSESSMENT.md` §2b. Already substantially
      satisfied by the §1y fusion-policy retrain (the primary decision
      path, 494/546 of benchmark decisions); the fixed fallback constants
      in `core/config.py` weren't touched since this run gave no evidence
      they need changing.
- [x] Rerun full benchmark, regression tests — done 2026-08-16, see
      `docs/ENGINEERING_ASSESSMENT.md` §2b. Also fixed a real, blocking
      config bug found along the way: `OLLAMA_MODEL` default was stale
      (`"mistral"`, never updated to `llama-guard3` for native/local
      runs). Two independent runs, bit-for-bit identical: Recall 60.59%
      (was 62.07%), Precision 83.11% (was 83.44%), F1 0.701 (was 0.712),
      **FPR unchanged at 7.29%**. Traced precisely: exactly 3 of 203
      attacks flipped from caught to missed, zero benign prompts affected
      — most likely the §1y fusion retrain, already validated
      out-of-fold on the larger suite with no regression there. Accepted
      and documented, not reverted — FPR (this project's hard constraint)
      is untouched, and the change closed a real taxonomy gap. 391 tests
      passed throughout.

## Phase 2 — Output Security (5/5 done)
Original estimate: ~15–25h

- [x] Secret detection (API keys, tokens) in LLM output — done 2026-08-21,
      see `docs/ENGINEERING_ASSESSMENT.md` §3a. New `core/secrets_detection.py`,
      8 vendor-specific patterns, hard BLOCK (no safe partial version of a
      leaked credential). Found and fixed a real bug in its own tests: two
      patterns used a capturing group, so `re.findall()` silently truncated
      the matched key before the preview logic ran.
- [x] System-prompt leakage detection — done 2026-08-21, see §3a. Opt-in
      (`system_prompt` field on `AssessRequest`/`AssessOutputRequest`) since
      Gatekeeper has no access to the caller's actual system prompt
      otherwise. Verbatim 40+-char substring check, not similarity — "about
      the same subject" and "contains the prompt" are different questions.
- [x] Unsafe output classification — confirmed already satisfied by the
      existing `output_judge` semantic-judge check, not rebuilt. Found and
      fixed a real, more urgent bug along the way: `output_judge` never got
      the Llama Guard protocol dispatch `semantic_judge` already has, so
      once `OLLAMA_MODEL`'s default became a Llama Guard variant (§2b),
      every output assessment would have failed closed to BLOCK
      unconditionally — a judge that blocks everything is not a working one.
- [x] Redaction on output, not just input — done 2026-08-21, see §3a.
      Behaviour change: PII in a response now redacts-and-continues
      (mirroring the input side) instead of blocking the entire response
      outright. `clean_response` added to both response schemas.
- [x] Output audit event distinct from input audit event — done 2026-08-21,
      see §3a. Found a real gap while building it: `/api/v1/assess_output`
      called no audit function at all — a BLOCKed response left zero audit
      trail. New `core/logger.py::log_output_event`, a genuinely separate
      record shape (not log_event with optional fields), wired into both
      endpoints.

27 new tests, 418 passed overall (up from 391).

Note: PII detection, block/release decision, and the wiring into the request
loop already existed (§1u) — this phase extended that, didn't start it.

## Phase 3 — Policy-as-Code (4/4 done)
Original estimate: ~15–25h

- [x] YAML/declarative policy format — done 2026-08-24, see
      `docs/ENGINEERING_ASSESSMENT.md` §3b. Added ALONGSIDE the existing
      JSON format (extension-dispatched), not a default-format migration —
      `policy_rules.json` is referenced by name in `docker-compose.yml`,
      `.dockerignore`, and `config/README.md`, so switching what a fresh
      deployment loads by default would need to touch all three for a
      readability win, not a capability one. New `policy_rules.yaml`
      worked example. `yaml.safe_load` only (not `yaml.load` — a policy
      file must not be a code-execution vector). Found PyYAML was an
      undeclared transitive dependency; declared it explicitly.
- [x] Validation step — done 2026-08-24, see §3b. New
      `core/policy.py::validate_policy_file` + `scripts/validate_policy.py`
      — reports EVERY problem in one pass, unlike the live loader which
      silently drops one malformed tenant to keep serving its siblings.
- [x] Simulation — done 2026-08-24, see §3b. New
      `scripts/simulate_policy.py`, replays real historical decisions from
      the audit log against a candidate policy with zero risk to live
      enforcement (`policy_decision()` gained an optional `store` param).
      Smoke-tested against this project's own real 6,801-record audit log.
- [x] Versioning and rollback — done 2026-08-24, see §3b. New
      `core/policy_versioning.py` + `scripts/manage_policy_versions.py`.
      Filesystem-based, deliberately not git-based — versions the LIVE
      deployed policy file, which git doesn't track in real time and which
      an operator may have no git access to at all.

32 new tests, 449 passed overall (up from 418).

## Phase 4 — Human Review (3/3 done)
Original estimate: ~10–15h

- [x] `REVIEW` as a distinct decision outcome — done 2026-08-24, see
      `docs/ENGINEERING_ASSESSMENT.md` §3c. Added to `core/policy.py`'s
      `VALID_ACTIONS`. Severity ordering for the combined call:
      `BLOCK > REVIEW > RESTRICT > ALLOW` — output-guard assessment is
      skipped when input is REVIEW, and REVIEW can never downgrade an
      already-BLOCK decision.
- [x] Review queue: review ID, reason, requester, risk, timestamp — done
      2026-08-24, see §3c. New `core/review_queue.py` — a single mutable
      JSON file (not the audit log's append-only convention), storing
      only the prompt's SHA-256 hash, never the raw text, matching
      `core/logger.py`'s own audit-record privacy convention exactly.
- [x] Approve/reject flow feeding back into the policy engine — done
      2026-08-24, see §3c. Three new endpoints: `GET /api/v1/review/
      {review_id}` (status, auth-only), `GET /api/v1/review` (pending
      list, `INTERNAL` capability), `POST /api/v1/review/{review_id}/
      resolve` (`INTERNAL` capability). "Feeds back" means the resolution
      becomes the request's final decision (APPROVED->ALLOW,
      REJECTED->BLOCK), retrieved by the original caller — does NOT
      retroactively affect future similar prompts (a deliberate privacy
      trade-off, not an oversight — see §3c's "what this does not close").

29 new tests, 478 passed overall (up from 449).

## Phase 5 — Real LLM Gateway (5/6 done — streaming closed by design decision, not built)
Original estimate: ~25–40h — **largest hardware risk on this machine**

- [x] Provider abstraction (Ollama / OpenAI-compatible / Anthropic-compatible)
      — done 2026-08-24, see `docs/ENGINEERING_ASSESSMENT.md` §3d. New
      `core/llm_providers.py`: one `LLMProvider.complete()` interface, three
      concrete backends, normalised `LLMResponse`. Deliberately stops there
      — no gateway endpoint, no streaming, no audit trail yet; see below
      for why those are separate items, not omissions.
- [x] Request forwarding + response interception — done 2026-08-24, see
      §3e. New `POST /api/v1/gateway/chat`: input guardrails gate the
      provider call (BLOCK/REVIEW never reach it), the response is run
      back through the Phase 2 output guardrails before being returned —
      a clean prompt does not guarantee a safe response, and the final
      decision reflects that.
- [x] Streaming support — CLOSED as an explicit architectural decision NOT
      to build it, not left undone by oversight. See §3f: this gateway's
      entire output-security value runs against the FULL assembled
      response (secrets/PII/toxicity/leakage checks are only meaningful
      over complete text), so real token-by-token streaming would hand
      back content before the one step that is this project's actual
      reason to exist. A "buffered pseudo-streaming" variant was
      considered and rejected — zero real latency benefit, real added
      complexity. Genuine streaming would need checkpointed re-inspection
      mid-stream, a real scope of future work, not a flag to flip.
- [x] Token accounting, model selection — done 2026-08-24, see §3f. New
      `core/token_quota.py::TokenQuotaTracker`, per-tenant daily quota
      (mirrors `rate_limit_rpm`'s override shape exactly), enforced
      against usage already recorded from PAST calls (a token cost can't
      be known before a call happens, so this can only ever gate the NEXT
      one). Model selection was already done in §3e.
- [x] Timeout/failure handling, fallback — done 2026-08-24, see §3f.
      `GATEWAY_FALLBACK_PROVIDERS` tries an ordered chain after the
      selected provider fails/times out — but ONLY when the caller left
      `provider` unset; an explicit choice is never second-guessed.
      Timeout half was already done in §3e.
- [x] Audit trail for the proxied call itself — done 2026-08-24, see §3e.
      New `core/logger.py::log_gateway_event`, a third distinct
      `event_type` (`"gateway_call"`) alongside input/output assessment
      events, joinable on `request_id`. Fires on both success and
      provider failure; never fires when input was BLOCK/REVIEW (no call
      was attempted).

57 new tests total across §3d–§3f, 536 passed overall (up from 478 at the
start of Phase 5). All mocked HTTP — no live network call or real API key
needed for any of them; the first genuine live provider call remains
unexercised (see §3f's "what this closes, honestly").

This is the point where Gatekeeper stops being a sidecar you call before/after
your own LLM call, and starts being infrastructure you route through — a
real architectural shift, not an incremental feature. Scope and benchmark
incrementally rather than building it in one push.

## Phase 6 — Tool / Agent Gateway
Original estimate: ~35–55h — the single largest new subsystem

- [ ] Tool registry + schemas
- [ ] Allow/deny, argument validation
- [ ] Risk levels, approval requirement
- [ ] Sandboxed demo tools
- [ ] Audit events
- [ ] MCP compatibility (explicitly deferred to after the above is solid)

## Phase 7 — UI Integration
Original estimate: ~40–65h combined — deferred until engine + gateways are solid

- [ ] Client UI: auth, dashboard, activity, privacy, approvals, protection settings
      (near-term subset already planned separately: general-user + operator
      views on top of the existing `gatekeeper-ui`)
- [ ] Developer UI: request inspector, detector signals, policy editor, model
      gateway view, tool gateway view, traces, benchmarks, logs

## Phase 8 — Production hardening (continuous, not a final phase)
Original estimate: ~20–35h

Runs alongside every phase above, same discipline used throughout this
project so far (real benchmarks before/after, CI gate per change, honest
"what this does not close" sections) rather than as a step done once at
the end.

- [ ] Integration / security / API contract tests per new subsystem
- [ ] Load testing
- [ ] Docker improvements, graceful failure handling
- [ ] Secrets management
- [ ] Structured logging / metrics / tracing extended to new subsystems
- [ ] Documentation, architecture diagrams, threat model updates

---

## Explicitly deferred (280–400h tier)

Not started, not scheduled — revisit only once Phases 1–8 are solid and
benchmarked: full MCP support, RBAC, API key management UI, organization
management, advanced quotas/cost controls, model routing/fallback,
distributed tracing, attack-campaign detection, compliance reporting,
policy simulation/versioning at enterprise depth, automated red-team
evaluation, proper frontend/backend separation as a shippable product.
