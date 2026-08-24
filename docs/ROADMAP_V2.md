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
      per the above) — SHIPPED 2026-08-24, merged to `main` via
      [PR #2](https://github.com/pavann19/Gatekeeper-AI-Infrastructure-and-Governance-Gateway/pull/2)
      (branch `phase1-german-gap-and-experiments`, deleted post-merge), see
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
      Verified live on real detectors, not just mocks. Three follow-ups
      filed, ALL SINCE CLOSED:
      [issue #3](https://github.com/pavann19/Gatekeeper-AI-Infrastructure-and-Governance-Gateway/issues/3)
      (German OFFENSIVE CONTENT, 0.584→0.742 AUC) — CLOSED 2026-08-24, see
      `docs/ENGINEERING_ASSESSMENT.md` §1ac: two off-the-shelf German
      toxicity detectors (`german_toxicity_eistakovskii`,
      `german_toxicity_ankekat`) added as an `eight_feature` tier,
      German-offensive AUC 0.597→0.741 with no decisive regression on any
      other axis (English actually improves, 0.941→0.950).
      [Issue #4](https://github.com/pavann19/Gatekeeper-AI-Infrastructure-and-Governance-Gateway/issues/4)
      (the multilingual head) — CLOSED 2026-08-24, see §1ad: leakage-free
      validated on a fully disjoint 3-way split (pooled AUC 0.923→0.945,
      German-injection recall@5%FPR 70.8%→93.1%, held-out data neither
      the head nor fusion had seen), wrapped as new
      `core/detectors.py::EmbeddingHeadDetector`, measured at 20ms warm
      per request, shipped as a new richest `nine_feature` tier.
      [Issue #5](https://github.com/pavann19/Gatekeeper-AI-Infrastructure-and-Governance-Gateway/issues/5)
      (`/api/v1/cache/flush` had no auth check) — CLOSED 2026-08-24, now
      requires `INTERNAL` capability, same bar as the review endpoints.
      German offensive-content detection is meaningfully better
      (0.584→0.784 across the full investigation) but remains the
      weakest of the three German-relevant numbers tracked in this
      document — no new issue opened for it, since "meaningfully
      improved, not perfect" describes most of this document's history
      before repeated passes moved the number further; revisit if a
      deployment needs it more than the now-94%+-recall injection case.
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

## Phase 6 — Tool / Agent Gateway (6/6 done)
Original estimate: ~35–55h — the single largest new subsystem

- [x] Tool registry + schemas — done 2026-08-24, see `core/tools.py`.
      `ToolSpec` (name, description, JSON-Schema-shaped `parameters`,
      `risk_level`, `capability_required`) and `ToolRegistry`
      (register/get/list, mirrors `core/detectors.py`'s registry
      pattern), plus `validate_arguments` for structural checks
      (required fields, declared types, enum membership — not full JSON
      Schema, not semantic validation). Deliberately stops there: no
      allow/deny wiring, no risk-based approval, no sandbox, no audit
      event, and no actual tools registered yet — this is schema-only
      groundwork, built before any of those so each later item is driven
      by what using this piece actually reveals is needed, not designed
      speculatively ahead of a real tool. 30 new tests (`tests/
      test_tools.py`), one real bug caught by its own test suite before
      merge (a JSON `true` would have silently passed an integer-typed
      argument check — `isinstance(True, int)` is `True` in Python).
      590 passed overall (up from 560).
- [x] Allow/deny, argument validation — done 2026-08-24, see `core/
      tools.py::check_tool_access`. Capability-based, minimum-privilege
      rank (`CAPABILITY_RANK`) reusing GENERAL/ELEVATED/INTERNAL, matching
      the ordering every policy actually shipped in this project follows
      in practice (documented as an observed convention, not a structural
      guarantee `core/policy.py`'s per-tenant JSON config could ever
      technically violate). Unrecognised capability values are denied,
      not defaulted to the lowest tier — fail closed. Argument validation
      covers structural checks only (required fields, types, enums);
      semantic/business-rule validation is inherently per-tool and
      deliberately deferred until a real tool exists to define it — that
      was never really a generic-gateway feature to begin with (the same
      scope OpenAI's own function-calling structural validation has). 6
      new tests, 596 passed overall (up from 590).
- [x] Risk levels, approval requirement — done 2026-08-24, see `core/
      tools.py::decide_tool_call`. Combines access control (already
      built) with a risk-based gate: HIGH-risk tools require REVIEW even
      for a caller whose capability already clears the access check —
      access answers "may this caller use this tool at all", approval
      answers "does THIS call need a human before it runs", and
      conflating them would let an INTERNAL caller's HIGH-risk call
      (e.g. a production delete) execute with nobody in the loop just
      because they're generally allowed to invoke the tool. Mirrors
      Phase 4's own REVIEW semantics exactly. Decision vocabulary
      (BLOCK/REVIEW/ALLOW) reuses `core/policy.py`'s action names minus
      RESTRICT, which has no obvious meaning for a tool call. At the time
      this item shipped, not yet wired into `core/review_queue.py` or an
      execution endpoint — see the phase-closing note below for where
      that wiring actually landed once `POST /api/v1/tools/call` existed
      to need it. 6 new tests, 602 passed overall (up from 596).
- [x] Sandboxed demo tools — done 2026-08-24, see `core/demo_tools.py`
      and `core/tools.py::execute_tool`. First real tools run through
      the full pipeline end to end (access control → structural
      validation → risk-based approval → execution), not just unit-
      tested against synthetic specs. Four tools chosen to exercise
      every branch `decide_tool_call` has: `demo.echo`/`demo.calculator.
      add` (LOW/GENERAL, trivial ALLOW), `demo.database.query` (MEDIUM/
      ELEVATED, capability gate matters), `demo.database.delete` (HIGH/
      INTERNAL, triggers REVIEW even for a caller who clears access
      control). "Sandboxed" means every handler touches only an
      in-memory fake dataset defined in the module — no filesystem,
      network, or real database — not a process-level sandbox
      (container/seccomp) enforced by `execute_tool` itself; the safety
      property lives in which handlers get registered, stated explicitly
      rather than implied. Not registered automatically on import —
      explicit opt-in via `register_demo_tools()`, so a production
      deployment doesn't get demo tools by default. `execute_tool` is
      also the first real enforcement point: a handler never runs for
      BLOCK or REVIEW, and a handler's own exception is reported
      distinctly from a security decision (the call WAS allowed to
      attempt running). 16 new tests, 628 passed overall (up from 602).
- [x] Audit events — done 2026-08-24, see `core/logger.py::log_tool_event`
      and its wiring into `core/tools.py::execute_tool`. A FOURTH
      distinct `event_type` (`"tool_call"`), alongside `log_event`'s
      `"input_assessment"`, `log_output_event`'s `"output_assessment"`,
      and `log_gateway_event`'s `"gateway_call"` — same reasoning as why
      those three stay separate: this answers a different question ("was
      this tool call allowed to run, and did it succeed?"). Emitted for
      EVERY decision, BLOCK and REVIEW included, not just successful
      calls — an unauthorized tool-call attempt is itself an auditable
      security event. Arguments are never logged verbatim, only their
      SHA-256 hash (`json.dumps(arguments, sort_keys=True)`), same
      privacy discipline `prompt_hash`/`response_hash` already apply —
      a tool argument can carry the same sensitive content a prompt can.
      `success`/`error` stay `None` for BLOCK/REVIEW rather than a
      misleading `False`, since a handler never ran for either. 15 new
      tests (`tests/test_tool_audit_events.py`), covering both the
      function directly (hash correctness, stable regardless of key
      order, no-hash-when-no-arguments) and the wiring (exactly one
      event per `execute_tool` call, on every branch). 643 passed
      overall (up from 628).
- [x] MCP compatibility — done 2026-08-24, see `core/mcp_compat.py`. Scoped
      honestly as PROTOCOL-SHAPE compatibility, not a transport server:
      `tool_spec_to_mcp`/`list_mcp_tools` produce MCP's `Tool` descriptor
      shape (close to a field rename, not real translation logic, because
      `ToolSpec.parameters` was JSON-Schema-shaped from the very first
      commit in this phase specifically so this step would be cheap), and
      `handle_mcp_tool_call` routes every call through `execute_tool`'s
      full pipeline (access, validation, risk, execution, audit)
      unchanged, reshaping the result into MCP's `CallToolResult`. A real
      stdio/SSE MCP server — the actual transport layer — is explicitly
      NOT built here: that is separate infrastructure with its own
      integration surface, belonging to whichever deployment actually
      needs to expose these tools over MCP, not to a compatibility layer
      built speculatively ahead of one. One stated, real protocol
      limitation: MCP's `CallToolResult` only distinguishes success from
      error, so Gatekeeper's BLOCK and REVIEW both become `isError: true`
      — the full decision is still completely audited underneath either
      way, only the MCP-facing response loses that distinction, because
      MCP's own shape has nowhere to put it. 14 new tests, 657 passed
      overall (up from 643).

**Phase 6 (Tool / Agent Gateway) is now complete, 6/6.** Registry and
schemas, allow/deny, risk-based approval, sandboxed demo tools, audit
events, and MCP-shape compatibility — built in that order specifically
so each item was driven by what using the previous one actually needed,
never speculatively ahead of a real caller.

**Closed out same day (2026-08-24), once questioned directly**: the two
gaps this phase's own writeup had left open — no live endpoint, and
REVIEW not wired to the real review queue — are now closed. New `POST
/api/v1/tools/call` in `api/main.py` gives `execute_tool` a real caller
with real `tenant`/`request_id` context, same auth/tenant/rate-limit
boundary as every other endpoint. A REVIEW decision now enqueues a real,
retrievable `ReviewRecord` via `core/review_queue.py::enqueue_review` —
verified end to end through the actual HTTP layer, not mocked: a real
`POST` for `demo.database.delete` produces a real PENDING review record,
resolvable via the existing Phase-4 `GET`/`POST /api/v1/review/*`
endpoints, no new review machinery built. `core/demo_tools.py`'s four
tools can now be registered at startup via `REGISTER_DEMO_TOOLS` (default
OFF — opt-in, same as the module always documented), so the endpoint has
something real to call without a deployment writing its own tools first.
12 new tests, 669 passed overall (up from 657).

**The real MCP transport server also closed out same day (2026-08-24)**,
once asked for directly. New `core/mcp_server.py`: JSON-RPC 2.0 over
newline-delimited stdio, hand-rolled rather than the official `mcp` SDK
— the SDK was tried first and its dependency tree bumped `starlette` to
a version incompatible with this project's pinned FastAPI, breaking
every API test module on contact; reverted immediately, confirmed clean
before writing any server code. `core/mcp_compat.py` already had every
piece of real logic (list tools, call a tool); this is only the framing
and method dispatch (`initialize`, `notifications/initialized`,
`tools/list`, `tools/call`) — genuinely less code than working around
the SDK conflict, and zero new dependencies. New `scripts/run_mcp_server.py`
CLI entrypoint, `--demo-tools` flag to make it runnable standalone. One
stated, deliberate scope boundary: MCP's stdio transport has no
per-call authentication, so the server runs at ONE fixed capability for
its whole process lifetime (`--capability`), the same trust-boundary
shape every real local MCP server already has — not a missing feature,
a property of the transport. 18 new tests (dispatch logic and the real
transport loop, including malformed JSON and an internal dispatch bug
both surviving without killing the session) plus a genuine subprocess
end-to-end smoke test (real stdin/stdout, not `io.StringIO`) confirming
`initialize` → `tools/list` → `tools/call` works over an actual child
process. 687 passed overall (up from 669).

**The first real (non-demo) tool also shipped same day (2026-08-24)**,
once asked for directly. New `core/real_tools.py::http.get` — an
outbound HTTP GET restricted to an operator-configured domain allowlist
(`TOOL_HTTP_GET_ALLOWED_DOMAINS`, empty by default: fail closed, the
tool is fully disabled until an operator explicitly lists at least one
hostname). Unlike the demo tools, this makes a real network call, so it
carries real security engineering most "add an HTTP tool" examples skip:
the hostname's OWN resolved IP is checked at call time and rejected if
private/loopback/link-local/reserved (DNS-rebinding defence — an
allowlisted domain's DNS could otherwise be repointed at `127.0.0.1` or
the cloud-metadata address `169.254.169.254` between the allowlist check
and the actual request), redirects are never followed
(`allow_redirects=False` — a 3xx response is otherwise a clean allowlist
bypass), and the response is both time-boxed and size-capped. A
disallowed URL is reported as an ALLOW decision with `error` set, not a
gateway-level BLOCK — `core/tools.py`'s own documented contract for
semantic/business-rule validation a handler performs, distinct from
access control the gateway itself decided (§ "Sandboxed demo tools"
above already drew this exact line). MEDIUM risk, ELEVATED capability.

23 new tests (mocked network/DNS, covering the allowlist, every SSRF
defence individually — private/loopback/cloud-metadata resolution,
DNS-failure-fails-closed, redirect handling, size cap — and the full
`execute_tool` pipeline), plus three genuine live checks against a real
server and real DNS: a real GET to `example.com` returning real content,
a real rejection of an unlisted domain, and a real rejection of
`localhost` via actual loopback resolution, not a mock standing in for
one. 710 passed overall (up from 687).

**Phase 6 is now completely closed — 6/6 roadmap items, both follow-up
items from the phase-completion review, no known gaps left unaddressed.**
The only thing left unbuilt is MCP's HTTP/SSE transport (a different,
per-call-authenticatable transport with its own auth story, not
attempted here) — genuinely out of scope until a deployment specifically
needs it, not a gap in what was asked for.

## Phase 7 — UI Integration
Original estimate: ~40–65h combined — deferred until engine + gateways are solid

- [x] Client UI (operator slice): human-review approval dashboard —
      `ui/review/index.html`, mounted (along with the rest of `ui/`) at
      `/ui/` by `api/main.py`. Started fresh rather than building on the
      pre-existing `ui/web_app.py` Streamlit prototype (found to reference a
      dead `/api/v1/update` endpoint removed back in Phase 1 §1z — legacy,
      disposable). Built as a single static page (system-font stack,
      card-based layout, restrained neutral palette with light/dark support
      via `prefers-color-scheme`, no build tooling) rather than Streamlit or
      a full SPA — the smallest real slice that still lets an operator poll
      `GET /api/v1/review`, and approve/reject via
      `POST /api/v1/review/{id}/resolve`, with an operator-supplied API key
      kept in `sessionStorage` (never persisted server-side). Verified live,
      end-to-end, against a real running instance: real INTERNAL API key
      issued via `scripts/manage_api_keys.py`, real reviews enqueued into a
      real (scratch) `ReviewQueue`, dashboard loaded in a real browser,
      approve/reject exercised against the real endpoints — confirmed the
      resulting `ReviewRecord`s carried the correct `status`,
      `final_decision`, and `reviewer` fields, not just a 200 response.
      (Originally at `ui/review_dashboard/`, renamed to `ui/review/` when
      the auth piece below needed a whole-`ui/`-tree static mount instead
      of a single-directory one, so the URL stayed `/ui/review/`.)
- [x] Client UI (auth): a dedicated sign-in page, `ui/login/index.html`,
      plus a new `GET /api/v1/whoami` endpoint (`api/main.py`,
      `api/schemas.py::WhoAmIResponse`) that resolves a pasted API key
      through the real `core.auth.resolve_principal` — the same function
      every enforcement decision uses — rather than a separate,
      potentially-divergent check. A key that fails resolves to a real 401,
      not a client-side format guess. On success the page redirects
      INTERNAL-capability keys straight into the review dashboard; other
      capabilities see their confirmed identity and an honest "no dashboard
      section built for this yet" message rather than a fake redirect.
      Shared `ui/shared/theme.css` (design tokens) and `ui/shared/auth.js`
      (`gkRequireAuth`, `gkSignOut`, session helpers) factor out what the
      login page and the review dashboard both need; the dashboard now
      gates itself through `gkRequireAuth()` on load instead of showing an
      inline key field, and gained a Sign Out control. Session lives only
      in `sessionStorage` (cleared on tab close, never persisted to disk).
      Verified live end-to-end in a real browser against a real running
      instance: a bogus key produces a real 401 and stays on the login
      page; a real GENERAL-capability key signs in and correctly reports no
      dashboard for it; a real INTERNAL-capability key signs in and
      auto-redirects into the review dashboard showing its own
      `key_id`/`tenant`; Sign Out clears the session and redirects to
      login; and a direct, unauthenticated navigation to the dashboard URL
      bounces straight back to login. Backend covered by
      `tests/test_whoami_endpoint.py` (6 tests: valid key, missing
      credential, unrecognised key, malformed header, no credential leakage
      in the response, and the exact minimal response shape).
- [x] Client UI (dashboard/activity): a real activity feed —
      `ui/activity/index.html`, backed by a new `GET /api/v1/activity`
      endpoint (`api/main.py`) and `core/activity.py`. Reads the SAME
      `audit.jsonl` every governance decision already writes to
      (`core/logger.py`'s four `log_*_event` functions) — this is a read
      view over the existing audit trail, not a new data source, and every
      event shown (input/output assessments, gateway calls, tool calls) is
      a record something real actually decided, not synthesized for
      display.
      `core/activity.py::get_recent_activity` tails the log from the end in
      fixed-size chunks rather than loading the whole file — the real
      `audit.jsonl` this was tested against is 14,000+ lines / 7MB+ and
      only growing — bounded by `MAX_BYTES_SCANNED` (20MB) so a filter
      matching almost nothing can't turn into an unbounded full-file scan;
      hitting that cap is reported back as `scan_truncated` rather than
      silently looking like "that's all there is." A pre-event_type legacy
      log line (from before this project's audit schema existed) is
      surfaced as `event_type: "legacy"` rather than dropped.
      Every authenticated caller sees their OWN tenant's activity by
      default (never overridable by a non-INTERNAL caller passing
      `?tenant=`, confirmed by test); INTERNAL may additionally request one
      other tenant or `?tenant=__all__` for everything, mirroring
      `list_reviews`' own cross-tenant reasoning. This is now the landing
      page after login for every capability (the review dashboard remains
      INTERNAL-only, reachable via a nav link `ui/shared/auth.js` now adds
      to both pages).
      Verified two ways: `tests/test_activity.py` (13 tests against real
      files on disk, including one that forces a tiny chunk size so a
      single JSON line is guaranteed to straddle multiple reads --
      catching a real infinite-loop bug in the first draft's retry logic
      where a small file that fits in one read chunk never set the
      "truncated" flag, so a filter matching almost nothing spun forever)
      and `tests/test_activity_endpoint.py` (8 tests over the real HTTP
      layer, tenant-scoping included) -- plus a live run in a real browser
      against the actual accumulated `audit.jsonl` (14,000+ real lines from
      this project's own dev/test history) and the actual `review_queue.json`
      (35 real pending reviews from earlier sessions), sign-in as both a
      GENERAL and an INTERNAL key, event-type filtering, the `__all__`
      cross-tenant view, and cross-navigation between Activity and Review
      Queue -- all read-only against that real data; nothing in it was
      approved/rejected/modified by this verification pass.
- [ ] Client UI (remainder): privacy, protection settings
- [x] Developer UI (request inspector + traces + detector signals):
      `ui/trace/index.html`, backed by `core.activity.find_by_request_id`
      and `GET /api/v1/activity/trace/{request_id}`. All three roadmap
      bullets collapse into one real capability: every audit line sharing
      a `request_id`, in chronological order, rendered with its FULL raw
      field set (not the activity feed's friendly one-line summary) --
      which for an `input_assessment` line means `semantic_score`,
      `symbolic_triggered`, `judge_invoked`, `domain_score`, `risk` are
      shown exactly as logged, i.e. this page IS the detector-signal view
      for a given request, not a separate feature. Same tenant-scoping
      rule as the activity feed (own tenant by default, INTERNAL may
      cross), open to every capability since inspecting your OWN request
      is not operator-only. Available to every capability via the shared
      nav; INTERNAL-only pages below add their own nav entries.
      `find_by_request_id` scans the full byte budget rather than
      stopping at a small count (a request_id is a sparse, scattered
      match, unlike "give me the last 50") -- covered by 6 new tests
      including a forced-tiny-chunk-size multi-chunk-reconstruction case.
- [x] Developer UI (model gateway view + tool gateway view):
      `ui/gateways/index.html`, backed by two new INTERNAL-only endpoints
      -- `GET /api/v1/gateway/providers` (real supported provider types
      from `core.llm_providers.list_provider_names`, plus the configured
      default) and `GET /api/v1/tools` (the real registered `ToolSpec`
      catalogue from the shared `ToolRegistry` -- name, JSON-Schema
      parameters, risk level, required capability). Each section also
      shows real recent activity (`GET /api/v1/logs` filtered by
      `gateway_call`/`tool_call`) alongside the static configuration, so
      the page answers both "what's configured" and "what's actually
      happening" for each gateway. 12 new tests.
- [x] Developer UI (logs): `ui/logs/index.html` and `GET /api/v1/logs` --
      the SAME `get_recent_activity` the activity feed uses, but
      INTERNAL-only, cross-tenant by default (the activity feed defaults
      to the caller's own tenant; this is the deliberately different,
      developer-facing default), raw JSON per line rather than the
      friendly card rendering. 7 new tests covering the capability gate
      and default-vs-scoped tenant behaviour.
- [x] Developer UI (benchmarks): `ui/benchmarks/index.html`, backed by
      `core.benchmarks.list_benchmark_runs` and `GET /api/v1/benchmarks`
      (INTERNAL-only). Reads this project's OWN real, already-committed
      benchmark evidence (`_evidence/benchmark_results_*.json` --
      accuracy/precision/recall/F1/FPR, confusion matrix, cold-vs-warm-
      cache latency, exactly as this project's benchmark scripts produced
      them) rather than re-deriving or synthesizing numbers. Deliberately
      scoped to only the `benchmark_results_*` shape -- `_evidence/`
      holds other real report types (calibration curves, detector
      comparisons) with genuinely different shapes not yet surfaced by
      this view. 10 new tests, including one asserting against the
      actual tracked evidence files on disk (a real trip wire if that
      file shape ever drifts from what this view assumes).

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
