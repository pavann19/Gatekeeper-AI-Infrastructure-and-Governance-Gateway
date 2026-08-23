"""
Prometheus instrumentation for the gateway.

WHY THIS EXISTS
---------------
`collect_semantic_signals` has been measuring per-stage latency all along —
`meta_intent_ms`, `faiss_threat_search_ms`, `domain_alignment_ms`, `fusion_ms`
— and then throwing it away into a response body that nothing aggregates. The
measurement was already there; only the exporter was missing. Everything
below either reuses a number the pipeline already computes or counts an event
it already distinguishes.

CARDINALITY IS THE WHOLE DESIGN PROBLEM
---------------------------------------
Prometheus stores one time series per distinct label combination, so a label
fed from unbounded input is not a metrics bug, it is an outage: memory grows
with the number of distinct values and the scrape eventually kills the
process. Three defences here, each guarding a place where unbounded values
could realistically get in:

1. `source` is checked against a closed set. It is a bounded set of literals
   in core/risk.py today, but "today" is the operative word — the guard means
   a future dynamic source value degrades to `other` instead of quietly
   multiplying series. `metrics_unknown_source_total` makes that degradation
   visible rather than silent.
2. `endpoint` uses the matched ROUTE TEMPLATE, never the raw path. Labelling
   by raw path would let anyone mint unlimited series by requesting random
   URLs; unmatched requests collapse to a single `unmatched` bucket.
3. `tenant` is operator-bounded — tenants come from the provisioned key store
   (core/auth.py), and callers cannot create them — so it is safe, but it is
   deliberately confined to its own low-dimensional counter rather than
   multiplied across the main one.

DECLARED LIMITATION
-------------------
These are per-process counters. Under a multi-worker server (gunicorn with
several workers) each worker exposes its own values, and a naive scrape sees
whichever worker answered. Correct multi-worker operation requires
`prometheus_client`'s multiprocess mode via a `PROMETHEUS_MULTIPROC_DIR`
shared directory. This is the same process-local caveat that applies to the
circuit breakers (§1k) and the rate limiter (§1q), and it is noted here for
the same reason: single-worker deployments are correct as-is, and multi-worker
ones need a deliberate step.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Bounded label vocabularies
# ---------------------------------------------------------------------------

# Every `source` core/risk.py can attach to a verdict. Kept as an explicit
# closed set rather than derived, so that adding a source is a conscious act
# that includes deciding whether it deserves its own series.
KNOWN_SOURCES = frozenset({
    "cache",
    "cache_locked_high",
    "clean_pass",
    "domain_guardrail",
    "educational_safe_harbor",
    "fusion_clean_pass",
    "fusion_educational_safe_harbor",
    "fusion_judge_pending",
    "fusion_threat_critical",
    "judge_failure_fail_closed",
    "judge_pending",
    "llama_guard_arbitration",
    "llama_guard_async_escalation",
    "llama_guard_override",
    "llama_guard_override_capped",
    "llama_guard_override_restricted",
    "semantic_judge",
    "semantic_judge_ambiguous",
    "semantic_judge_override",
    "semantic_judge_override_capped",
    "semantic_judge_override_restricted",
    "semantic_meta_intent",
    "fast_path_meta_intent",
    "fast_path_anchor_critical",
    "symbolic_rule",
    "vector_threat_critical",
    "mock",          # unit tests and harnesses
    "unknown",
})

# The per-stage timings collect_semantic_signals already records, mapped from
# the `details` key it writes to the label value exported.
STAGE_KEYS = {
    "meta_intent_ms": "meta_intent",
    "faiss_threat_search_ms": "faiss_threat_search",
    "domain_alignment_ms": "domain_alignment",
    "fusion_ms": "fusion",
}

# Stage timings span three orders of magnitude — sub-millisecond symbolic
# checks through multi-second judge calls (§1p measured p99 ~19s end to end),
# so the default buckets (which top out at 10s) would lump the entire tail
# into +Inf and make the worst case invisible. These are chosen to keep
# resolution where the decisions are: single-digit ms for the fast path,
# and real buckets past 10s for the judge.
_LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0,
)


def safe_source(source) -> str:
    """
    Constrains `source` to the known set, counting anything unexpected.

    Returns `other` for unrecognised values rather than passing them through,
    because an unbounded label value is the one metrics mistake that takes the
    process down rather than merely reporting badly.
    """
    if source in KNOWN_SOURCES:
        return source
    unknown_source_total.labels(source=_truncate(str(source))).inc()
    return "other"


def _truncate(value: str, limit: int = 40) -> str:
    return value if len(value) <= limit else value[:limit]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

assessments_total = Counter(
    "gatekeeper_assessments_total",
    "Completed prompt assessments, by outcome and what decided it.",
    ["decision", "risk_level", "source"],
)

tenant_assessments_total = Counter(
    "gatekeeper_tenant_assessments_total",
    "Completed prompt assessments per tenant. Kept separate from the main "
    "counter to avoid multiplying tenant count across every other dimension.",
    ["tenant", "decision"],
)

stage_duration_seconds = Histogram(
    "gatekeeper_stage_duration_seconds",
    "Per-stage assessment latency, from the timings the pipeline already "
    "measures.",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)

request_duration_seconds = Histogram(
    "gatekeeper_request_duration_seconds",
    "End-to-end HTTP request latency, by matched route template.",
    ["endpoint", "method", "status"],
    buckets=_LATENCY_BUCKETS,
)

assessments_in_flight = Gauge(
    "gatekeeper_assessments_in_flight",
    "Assessments currently executing or queued for the bounded worker pool. "
    "Sustained values at or above ASSESS_MAX_CONCURRENCY mean requests are "
    "queueing and the timeout is the next thing that will fire.",
)

rate_limited_total = Counter(
    "gatekeeper_rate_limited_total",
    "Requests rejected with 429, split by whether the caller was "
    "authenticated — anonymous overage is ordinary, authenticated overage "
    "usually means a client bug worth chasing.",
    ["authenticated"],
)

assessment_timeouts_total = Counter(
    "gatekeeper_assessment_timeouts_total",
    "Assessments abandoned at the deadline and answered with 503.",
    ["endpoint"],
)

judge_invocations_total = Counter(
    "gatekeeper_judge_invocations_total",
    "Ambiguous-zone arbitrations that actually reached a judge backend.",
)

# --- Phase 5 (Real LLM Gateway): the proxied call, kept separate from the
# assessment metrics above -- a slow or failing external provider is a
# different failure mode than a slow local assessment, and conflating them
# would hide which one is actually degraded.
gateway_calls_in_flight = Gauge(
    "gatekeeper_gateway_calls_in_flight",
    "Proxied LLM calls currently executing or queued for the gateway's "
    "bounded worker pool. Sustained values at or above "
    "GATEWAY_MAX_CONCURRENCY mean calls are queueing.",
)

gateway_call_total = Counter(
    "gatekeeper_gateway_call_total",
    "Proxied LLM calls, by provider and outcome (success/failure/timeout/blocked).",
    ["provider", "outcome"],
)

# Tenant is safe as its own low-dimension label here for the same reason
# tenant_assessments_total is kept separate from assessments_total above:
# tenant_id is operator-provisioned (core/tenancy.py), not caller-chosen.
gateway_tokens_total = Counter(
    "gatekeeper_gateway_tokens_total",
    "Tokens consumed via the gateway's proxied calls, by tenant. Only counts "
    "calls whose provider actually reported usage (core/token_quota.py's "
    "extract_total_tokens) -- a provider that doesn't report usage (Ollama) "
    "contributes zero here, not an estimate.",
    ["tenant"],
)

gateway_quota_rejections_total = Counter(
    "gatekeeper_gateway_quota_rejections_total",
    "Gateway calls rejected with 429 because the tenant's daily token quota "
    "was already exhausted BEFORE this call, by tenant.",
    ["tenant"],
)

circuit_breaker_open = Gauge(
    "gatekeeper_circuit_breaker_open",
    "1 while a judge backend's circuit breaker is open (failing fast).",
    ["backend"],
)

unknown_source_total = Counter(
    "gatekeeper_metrics_unknown_source_total",
    "Verdict sources not in the known set, collapsed to 'other'. Non-zero "
    "means core/risk.py grew a source that core/metrics.py has not been told "
    "about — the metric is degraded, not the gateway.",
    ["source"],
)


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def record_assessment(decision: str, risk_level: str, details: dict, tenant: str) -> None:
    """
    Records one completed assessment: the outcome, the tenant, and every
    per-stage timing the pipeline happened to measure for it.

    Tolerant by construction. Metrics are observability, not control flow — a
    malformed or partial `details` dict (a cache hit carries no stage timings,
    for instance) must degrade the metric, never the request. Nothing here
    raises on missing or non-numeric values.
    """
    source = safe_source(details.get("source", "unknown"))
    assessments_total.labels(
        decision=decision, risk_level=risk_level, source=source
    ).inc()
    tenant_assessments_total.labels(tenant=tenant, decision=decision).inc()

    for key, stage in STAGE_KEYS.items():
        value = details.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            stage_duration_seconds.labels(stage=stage).observe(value / 1000.0)

    if details.get("judge_invoked"):
        judge_invocations_total.inc()


def refresh_circuit_breaker_gauges() -> None:
    """
    Samples breaker state at scrape time rather than tracking transitions.

    A breaker opens and closes inside core/circuit_breaker.py without emitting
    an event, and adding callbacks there purely for metrics would couple a
    safety-critical component to an observability one. Reading the state when
    asked is both simpler and impossible to get out of sync.
    """
    from core.circuit_breaker import llama_guard_breaker, ollama_judge_breaker

    for breaker in (ollama_judge_breaker, llama_guard_breaker):
        # Read the flag directly: is_open() has a side effect — it transitions
        # a cooled-down breaker into its half-open probe. A scrape must never
        # change the behaviour of the thing it is measuring.
        with breaker._lock:
            is_open = breaker._opened_at is not None
        circuit_breaker_open.labels(backend=breaker.name).set(1 if is_open else 0)
