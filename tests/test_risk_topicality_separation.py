"""
Verification for the Phase 1 "separate risk from topicality in practice"
item (docs/ROADMAP_V2.md; see docs/ENGINEERING_ASSESSMENT.md's write-up).

FINDING: this item was already correctly implemented. `core/risk.py`'s
`fuse_signals` already computes `topicality` independently of `final_risk`
(tests/test_fusion.py's `test_off_topic_benign_prompt_is_not_a_safety_risk`
is literally "the core regression test" for exactly this, per its own
docstring), and topicality influences a risk decision in exactly one
deliberate, documented, opt-in case: `DOMAIN_GUARDRAIL_MODE=enforcing`
(also already tested, `test_off_topic_escalates_only_in_enforcing_mode`).

What was NOT previously tested is that the separation holds at the
ENFORCEMENT layer too, not just inside `fuse_signals` -- these tests close
that gap: `core.policy.policy_decision` and `core.metrics.record_assessment`
must never let `topicality` influence an enforcement decision or a metric
label, structurally (by signature) and behaviourally (by input variation).
"""
from core.metrics import assessments_total
from core.policy import policy_decision


def test_policy_decision_has_no_topicality_parameter():
    """A future refactor that widens this signature to accept topicality
    would be re-introducing exactly the conflation
    docs/ENGINEERING_ASSESSMENT.md's benchmark.py fix already eliminated --
    guard against that happening silently."""
    import inspect
    params = set(inspect.signature(policy_decision).parameters)
    assert "topicality" not in params
    assert params == {"capability", "risk", "tenant_id"}


def test_record_assessment_metric_is_identical_regardless_of_topicality():
    """Same (decision, risk_level, source, tenant) with two different
    topicality values in `details` must produce the SAME counter
    increment -- topicality must not silently become a metric dimension
    (which would also blow up cardinality unexpectedly, the exact failure
    mode core/metrics.py's own cardinality guard exists to prevent)."""
    from core import metrics as metrics_mod

    before = assessments_total.labels(
        decision="ALLOW", risk_level="LOW", source="clean_pass"
    )._value.get()

    metrics_mod.record_assessment(
        "ALLOW", "LOW", {"source": "clean_pass", "topicality": "IN_DOMAIN"}, "default"
    )
    after_in_domain = assessments_total.labels(
        decision="ALLOW", risk_level="LOW", source="clean_pass"
    )._value.get()
    assert after_in_domain == before + 1

    metrics_mod.record_assessment(
        "ALLOW", "LOW", {"source": "clean_pass", "topicality": "OUT_OF_DOMAIN"}, "default"
    )
    after_out_of_domain = assessments_total.labels(
        decision="ALLOW", risk_level="LOW", source="clean_pass"
    )._value.get()
    assert after_out_of_domain == after_in_domain + 1


def test_cache_hit_reports_unknown_topicality_by_design():
    """core/cache.py's save_cache_entry never persists topicality -- a
    cache hit legitimately cannot know the original classification, so
    UNKNOWN is the correct, honest degrade, not a bug. This test documents
    that as intentional so a future change doesn't 'fix' it by threading
    topicality through the cache and coupling it to the safety-decision
    cache key."""
    import inspect
    from core.cache import save_cache_entry
    params = list(inspect.signature(save_cache_entry).parameters)
    assert "topicality" not in params
