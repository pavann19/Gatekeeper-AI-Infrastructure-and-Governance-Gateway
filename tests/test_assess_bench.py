from benchmarks.load.assess_bench import (
    _judge_histogram_snapshot,
    _parse_prom_sample,
)


PROMETHEUS_TEXT = """
# HELP gatekeeper_stage_duration_seconds Per-stage latency.
gatekeeper_stage_duration_seconds_count{stage="fusion"} 11
gatekeeper_stage_duration_seconds_sum{stage="fusion"} 1.25
gatekeeper_stage_duration_seconds_count{stage="judge"} 3
gatekeeper_stage_duration_seconds_sum{stage="judge"} 4.5
"""


def test_prometheus_sample_selects_exact_judge_series():
    assert _parse_prom_sample(
        PROMETHEUS_TEXT,
        "gatekeeper_stage_duration_seconds_count",
        {"stage": "judge"},
    ) == 3.0


def test_judge_histogram_snapshot_keeps_count_and_duration_separate():
    assert _judge_histogram_snapshot(PROMETHEUS_TEXT) == (3.0, 4.5)


def test_judge_histogram_snapshot_is_absent_when_metrics_are_unavailable():
    assert _judge_histogram_snapshot(None) is None
    assert _judge_histogram_snapshot("unrelated_metric 1\n") is None
