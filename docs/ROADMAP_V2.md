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

## Phase 1 — Strengthen the existing security engine (in progress)
Original estimate: ~20–30h

- [x] Per-class risk vector groundwork identified (`class_scores` already
      computed in `core/fusion.py`, not yet surfaced) — **starting now**
- [ ] Surface per-class risk vector in `details` / API response
- [ ] Multilingual encoder (German recall gap documented in the engineering
      assessment — AUC 0.890 overall, German notably weaker)
- [ ] Clean threat taxonomy
- [ ] Separate `risk` from `topicality` (topicality field already exists in
      the schema — confirm it's actually independent in practice, not just
      in name)
- [ ] Investigate/remove dead dynamic threat feed
- [ ] Recalibrate thresholds
- [ ] Rerun full benchmark, regression tests

## Phase 2 — Output Security
Original estimate: ~15–25h

- [ ] Secret detection (API keys, tokens) in LLM output
- [ ] System-prompt leakage detection
- [ ] Unsafe output classification (extends existing `core/output_guardrails.py`)
- [ ] Redaction on output, not just input
- [ ] Output audit event distinct from input audit event

Note: PII detection, block/release decision, and the wiring into the request
loop already exist (§1u) — this phase extends that, doesn't start it.

## Phase 3 — Policy-as-Code
Original estimate: ~15–25h

- [ ] YAML/declarative policy format replacing the current
      `policy_rules.json` capability/risk matrix
- [ ] Validation step
- [ ] Simulation (dry-run a policy change against historical traffic before
      deploying it)
- [ ] Versioning and rollback

## Phase 4 — Human Review
Original estimate: ~10–15h

- [ ] `REVIEW` as a distinct decision outcome (flagged open since the
      original Phase 0 V2 audit)
- [ ] Review queue: review ID, reason, requester, risk, timestamp
- [ ] Approve/reject flow feeding back into the policy engine

## Phase 5 — Real LLM Gateway
Original estimate: ~25–40h — **largest hardware risk on this machine**

- [ ] Provider abstraction (Ollama / OpenAI-compatible / Anthropic-compatible)
- [ ] Request forwarding + response interception
- [ ] Streaming support
- [ ] Token accounting, model selection
- [ ] Timeout/failure handling, fallback
- [ ] Audit trail for the proxied call itself

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
