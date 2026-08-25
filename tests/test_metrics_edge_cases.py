"""
Edge-case coverage for core/metrics.py's recording helpers and label
vocabularies.

This file is deliberately additive to tests/test_fast_path_cascade.py,
tests/test_policy_editor_endpoint.py, and tests/test_risk_topicality_separation.py
(the existing referrers to core.metrics) — it does not repeat what those files
already exercise incidentally through their own endpoints/pipeline calls.
Every test here asserts a real numeric value read off `._value.get()` /
`._sum.get()` / `._count.get()` or a real label set, never just "no exception".
"""
import threading

import pytest

from core import metrics
from core.circuit_breaker import CircuitBreaker


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def _hist_count(hist, **labels):
    # Histogram child objects expose no public `_count` attribute. Each entry
    # in `_buckets` stores a PER-BUCKET (non-cumulative) tally internally --
    # prometheus_client only cumulates them at collect()/export time -- so the
    # total observation count is the sum across all buckets, not the last one.
    return sum(b.get() for b in hist.labels(**labels)._buckets)


def _hist_sum(hist, **labels):
    return hist.labels(**labels)._sum.get()


# ---------------------------------------------------------------------------
# safe_source: bounded label vocabulary
# ---------------------------------------------------------------------------

def test_safe_source_known_value_passes_through_unchanged():
    assert metrics.safe_source("symbolic_rule") == "symbolic_rule"


def test_safe_source_unknown_value_collapses_to_other_and_counts_it():
    before = _counter_value(metrics.unknown_source_total, source="totally_new_source")
    result = metrics.safe_source("totally_new_source")
    after = _counter_value(metrics.unknown_source_total, source="totally_new_source")
    assert result == "other"
    assert after == before + 1


def test_safe_source_truncates_long_unknown_values_to_40_chars():
    long_value = "x" * 100
    metrics.safe_source(long_value)
    truncated_label = long_value[:40]
    # The counter must have been recorded under the truncated label, not the
    # full 100-char value -- an untruncated label is exactly the unbounded-
    # cardinality bug this function exists to prevent.
    value = _counter_value(metrics.unknown_source_total, source=truncated_label)
    assert value >= 1


def test_safe_source_none_is_treated_as_unknown():
    result = metrics.safe_source(None)
    assert result == "other"


# ---------------------------------------------------------------------------
# record_assessment: every decision / risk_level / source path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decision", ["ALLOW", "BLOCK", "REVIEW", "RESTRICT"])
def test_record_assessment_increments_correct_decision_label(decision):
    before = _counter_value(
        metrics.assessments_total, decision=decision, risk_level="HIGH", source="mock"
    )
    metrics.record_assessment(decision, "HIGH", {"source": "mock"}, tenant="acme")
    after = _counter_value(
        metrics.assessments_total, decision=decision, risk_level="HIGH", source="mock"
    )
    assert after == before + 1


@pytest.mark.parametrize("risk_level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
def test_record_assessment_increments_correct_risk_level_label(risk_level):
    before = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level=risk_level, source="mock"
    )
    metrics.record_assessment("ALLOW", risk_level, {"source": "mock"}, tenant="acme")
    after = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level=risk_level, source="mock"
    )
    assert after == before + 1


def test_record_assessment_does_not_cross_contaminate_other_label_combos():
    before_target = _counter_value(
        metrics.assessments_total, decision="BLOCK", risk_level="CRITICAL", source="symbolic_rule"
    )
    before_other = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="clean_pass"
    )
    metrics.record_assessment(
        "BLOCK", "CRITICAL", {"source": "symbolic_rule"}, tenant="acme"
    )
    after_target = _counter_value(
        metrics.assessments_total, decision="BLOCK", risk_level="CRITICAL", source="symbolic_rule"
    )
    after_other = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="clean_pass"
    )
    assert after_target == before_target + 1
    assert after_other == before_other  # untouched


def test_record_assessment_increments_tenant_counter_with_correct_labels():
    before = _counter_value(
        metrics.tenant_assessments_total, tenant="tenant-x", decision="REVIEW"
    )
    metrics.record_assessment("REVIEW", "MEDIUM", {"source": "mock"}, tenant="tenant-x")
    after = _counter_value(
        metrics.tenant_assessments_total, tenant="tenant-x", decision="REVIEW"
    )
    assert after == before + 1


def test_record_assessment_maps_unknown_source_through_safe_source():
    before = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="other"
    )
    metrics.record_assessment(
        "ALLOW", "LOW", {"source": "some_brand_new_source_xyz"}, tenant="acme"
    )
    after = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="other"
    )
    assert after == before + 1


def test_record_assessment_defaults_missing_source_to_unknown():
    # "unknown" is itself in KNOWN_SOURCES, so a missing `source` key passes
    # through safe_source() unchanged rather than collapsing to "other".
    before = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="unknown"
    )
    metrics.record_assessment("ALLOW", "LOW", {}, tenant="acme")
    after = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="unknown"
    )
    assert after == before + 1


def test_record_assessment_observes_stage_durations_with_correct_stage_labels():
    before_meta = _hist_count(metrics.stage_duration_seconds, stage="meta_intent")
    before_meta_sum = _hist_sum(metrics.stage_duration_seconds, stage="meta_intent")
    before_fusion = _hist_count(metrics.stage_duration_seconds, stage="fusion")

    metrics.record_assessment(
        "ALLOW",
        "LOW",
        {
            "source": "mock",
            "meta_intent_ms": 12.5,
            "fusion_ms": 3.0,
        },
        tenant="acme",
    )

    after_meta = _hist_count(metrics.stage_duration_seconds, stage="meta_intent")
    after_meta_sum = _hist_sum(metrics.stage_duration_seconds, stage="meta_intent")
    after_fusion = _hist_count(metrics.stage_duration_seconds, stage="fusion")

    assert after_meta == before_meta + 1
    # observed in seconds, not ms -- 12.5ms -> 0.0125s
    assert after_meta_sum == pytest.approx(before_meta_sum + 0.0125)
    assert after_fusion == before_fusion + 1


def test_record_assessment_skips_stage_timing_absent_from_details():
    before = _hist_count(metrics.stage_duration_seconds, stage="domain_alignment")
    metrics.record_assessment(
        "ALLOW", "LOW", {"source": "mock", "meta_intent_ms": 1.0}, tenant="acme"
    )
    after = _hist_count(metrics.stage_duration_seconds, stage="domain_alignment")
    assert after == before  # untouched: domain_alignment_ms was never in details


def test_record_assessment_ignores_non_numeric_stage_timing():
    before = _hist_count(metrics.stage_duration_seconds, stage="fusion")
    metrics.record_assessment(
        "ALLOW", "LOW", {"source": "mock", "fusion_ms": "not-a-number"}, tenant="acme"
    )
    after = _hist_count(metrics.stage_duration_seconds, stage="fusion")
    assert after == before  # malformed value must not raise or be recorded


def test_record_assessment_ignores_negative_stage_timing():
    before = _hist_count(metrics.stage_duration_seconds, stage="faiss_threat_search")
    metrics.record_assessment(
        "ALLOW", "LOW", {"source": "mock", "faiss_threat_search_ms": -5.0}, tenant="acme"
    )
    after = _hist_count(metrics.stage_duration_seconds, stage="faiss_threat_search")
    assert after == before


def test_record_assessment_increments_judge_invocations_when_flagged():
    before = metrics.judge_invocations_total._value.get()
    metrics.record_assessment(
        "REVIEW", "HIGH", {"source": "mock", "judge_invoked": True}, tenant="acme"
    )
    after = metrics.judge_invocations_total._value.get()
    assert after == before + 1


def test_record_assessment_does_not_increment_judge_invocations_when_absent():
    before = metrics.judge_invocations_total._value.get()
    metrics.record_assessment("ALLOW", "LOW", {"source": "mock"}, tenant="acme")
    after = metrics.judge_invocations_total._value.get()
    assert after == before


def test_record_assessment_tolerates_none_details_like_missing_keys():
    # details.get(...) is used throughout; an empty dict must not raise, and
    # the counters must still move for the decision/risk_level/tenant labels.
    before = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="other"
    )
    metrics.record_assessment("ALLOW", "LOW", {"source": None}, tenant="acme")
    after = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="other"
    )
    assert after == before + 1


# ---------------------------------------------------------------------------
# refresh_circuit_breaker_gauges: must read state without mutating it
# ---------------------------------------------------------------------------

def test_refresh_sets_gauge_to_zero_when_breaker_closed():
    from core.circuit_breaker import ollama_judge_breaker

    ollama_judge_breaker.reset()
    metrics.refresh_circuit_breaker_gauges()
    value = metrics.circuit_breaker_open.labels(backend="ollama_judge")._value.get()
    assert value == 0


def test_refresh_sets_gauge_to_one_when_breaker_open_and_does_not_mutate_it():
    from core.circuit_breaker import ollama_judge_breaker

    ollama_judge_breaker.reset()
    for _ in range(ollama_judge_breaker.failure_threshold):
        ollama_judge_breaker.record_failure()
    assert ollama_judge_breaker._opened_at is not None  # sanity: actually open

    opened_at_before = ollama_judge_breaker._opened_at
    metrics.refresh_circuit_breaker_gauges()
    value = metrics.circuit_breaker_open.labels(backend="ollama_judge")._value.get()

    assert value == 1
    # The critical regression this guards: refresh must read `_opened_at`
    # directly, never call is_open(), because is_open() has the side effect
    # of transitioning a cooled-down breaker into a half-open probe. Calling
    # it from a scrape would silently alter breaker behavior.
    assert ollama_judge_breaker._opened_at == opened_at_before

    ollama_judge_breaker.reset()


def test_refresh_reads_both_known_backends_independently():
    from core.circuit_breaker import llama_guard_breaker, ollama_judge_breaker

    ollama_judge_breaker.reset()
    llama_guard_breaker.reset()
    for _ in range(llama_guard_breaker.failure_threshold):
        llama_guard_breaker.record_failure()

    metrics.refresh_circuit_breaker_gauges()

    ollama_value = metrics.circuit_breaker_open.labels(backend="ollama_judge")._value.get()
    llama_value = metrics.circuit_breaker_open.labels(backend="llama_guard")._value.get()

    assert ollama_value == 0
    assert llama_value == 1

    llama_guard_breaker.reset()


def test_refresh_never_calls_is_open_on_the_singletons(monkeypatch):
    from core.circuit_breaker import llama_guard_breaker, ollama_judge_breaker

    def _boom(self=None):
        raise AssertionError(
            "refresh_circuit_breaker_gauges must not call is_open() -- "
            "it has the side effect of transitioning a cooled breaker into "
            "a half-open probe, which a metrics scrape must never trigger."
        )

    monkeypatch.setattr(ollama_judge_breaker, "is_open", _boom)
    monkeypatch.setattr(llama_guard_breaker, "is_open", _boom)

    # Must not raise.
    metrics.refresh_circuit_breaker_gauges()


def test_refresh_with_a_fresh_breaker_instance_reads_lock_protected_state():
    # Exercises the exact attribute path refresh_circuit_breaker_gauges uses
    # (breaker._lock / breaker._opened_at / breaker.name) against a breaker
    # object it did not create, to pin the contract independent of the
    # module-level singletons' current state.
    fresh = CircuitBreaker("scratch_backend", failure_threshold=1, cooldown_seconds=999)
    fresh.record_failure()
    with fresh._lock:
        is_open = fresh._opened_at is not None
    metrics.circuit_breaker_open.labels(backend=fresh.name).set(1 if is_open else 0)
    value = metrics.circuit_breaker_open.labels(backend="scratch_backend")._value.get()
    assert value == 1


# ---------------------------------------------------------------------------
# Recording is tolerant: metrics must never be the reason a request fails.
# core/metrics.py's own docstring on record_assessment states nothing in it
# raises on missing/non-numeric values -- verify that contract directly for
# the malformed-details shapes a real cache-hit / partial pipeline run could
# produce, without any exception propagating out.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "details",
    [
        {},
        {"source": "unknown"},
        {"meta_intent_ms": None},
        {"fusion_ms": []},
        {"domain_alignment_ms": {"nested": "dict"}},
        {"faiss_threat_search_ms": float("nan")},
        {"judge_invoked": "not-a-bool-but-truthy"},
        {"source": 12345},
    ],
)
def test_record_assessment_never_raises_on_malformed_details(details):
    # Should complete without raising for any of these malformed shapes.
    metrics.record_assessment("ALLOW", "LOW", details, tenant="acme")


def test_record_assessment_is_thread_safe_for_concurrent_callers():
    # Prometheus client counters are thread-safe internally; this pins that
    # concurrent record_assessment calls land every increment (no lost
    # updates), which matters because the gateway calls this from multiple
    # request-handling threads.
    before = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="mock"
    )
    n_threads = 20

    def _record():
        metrics.record_assessment("ALLOW", "LOW", {"source": "mock"}, tenant="acme")

    threads = [threading.Thread(target=_record) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    after = _counter_value(
        metrics.assessments_total, decision="ALLOW", risk_level="LOW", source="mock"
    )
    assert after == before + n_threads
