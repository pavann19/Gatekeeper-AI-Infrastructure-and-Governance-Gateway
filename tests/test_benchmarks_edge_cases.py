"""
Additional edge-case coverage for core/benchmarks.py, complementing
tests/test_benchmarks.py. Focuses on: realistic multi-file discovery
against a tmp_path fixture shaped like this project's real
_evidence/benchmark_results_*.json files, mixed valid/malformed
listings, and the defensive non-dict trip wire from the loader's own
side (as opposed to the API-response-shape side already covered
elsewhere).
"""
import json
import os

from core.benchmarks import list_benchmark_runs

# Trimmed but structurally realistic shape, modeled directly on the real
# tracked file _evidence/benchmark_results_run1_noisy.json.
REALISTIC_RUN = {
    "valid": True,
    "invalid_reason": None,
    "config": {
        "domain_guardrail_mode": "off",
        "semantic_threshold_high": 0.48,
        "semantic_threshold_medium": 0.3,
        "meta_intent_threshold": 0.3,
        "domain_threshold": 0.22,
        "embedding_model": "all-mpnet-base-v2",
        "judge_model": "llama-guard3",
    },
    "dataset": {"name": "deepset/prompt-injections", "n": 546},
    "cold": {
        "run_name": "Cold Cache (uncached, FAISS)",
        "avg_latency_ms": 1169.64,
        "p50_latency_ms": 1058.25,
        "p95_latency_ms": 1923.23,
        "p99_latency_ms": 2827.32,
        "metrics_operational": {
            "mode": "operational (HIGH or MEDIUM)",
            "accuracy": 0.807,
            "precision": 0.831,
            "recall_tpr": 0.605,
            "f1": 0.700,
            "fpr": 0.072,
            "confusion_matrix": {"TP": 123, "FP": 25, "TN": 318, "FN": 80},
        },
        "risk_distribution": {"LOW": 398, "HIGH": 148},
        "judge_invocations": 36,
    },
    "warm": {
        "run_name": "Warm Cache (LRU + FAISS)",
        "avg_latency_ms": 68.22,
        "p50_latency_ms": 56.61,
        "p95_latency_ms": 121.48,
        "p99_latency_ms": 251.73,
        "metrics_operational": {
            "mode": "operational (HIGH or MEDIUM)",
            "accuracy": 0.807,
            "precision": 0.831,
            "recall_tpr": 0.605,
            "f1": 0.700,
            "fpr": 0.072,
            "confusion_matrix": {"TP": 123, "FP": 25, "TN": 318, "FN": 80},
        },
        "risk_distribution": {"LOW": 398, "HIGH": 148},
        "judge_invocations": 0,
    },
    "speedup": 17.14,
}


def _write_run(path, name, overrides=None, mtime_offset=None):
    content = dict(REALISTIC_RUN)
    if overrides:
        content.update(overrides)
    file_path = path / name
    file_path.write_text(json.dumps(content), encoding="utf-8")
    if mtime_offset is not None:
        t = os.path.getmtime(file_path) + mtime_offset
        os.utime(file_path, (t, t))
    return file_path


def test_multiple_realistic_files_all_discovered_with_full_content(tmp_path):
    _write_run(tmp_path, "benchmark_results_deepset.json", {"speedup": 17.14})
    _write_run(
        tmp_path,
        "benchmark_results_german.json",
        {"dataset": {"name": "german_toxicity", "n": 200}, "speedup": 5.0},
    )
    result = list_benchmark_runs(directory=str(tmp_path))
    assert result["errors"] == []
    assert len(result["runs"]) == 2

    by_name = {r["_filename"]: r for r in result["runs"]}
    deepset = by_name["benchmark_results_deepset.json"]
    german = by_name["benchmark_results_german.json"]

    # Real nested content survives the read intact, not just top-level keys.
    assert deepset["config"]["judge_model"] == "llama-guard3"
    assert deepset["cold"]["metrics_operational"]["confusion_matrix"] == {
        "TP": 123, "FP": 25, "TN": 318, "FN": 80,
    }
    assert deepset["warm"]["run_name"] == "Warm Cache (LRU + FAISS)"
    assert deepset["speedup"] == 17.14

    assert german["dataset"] == {"name": "german_toxicity", "n": 200}
    assert german["speedup"] == 5.0


def test_ordering_is_newest_first_across_several_realistic_files(tmp_path):
    _write_run(tmp_path, "benchmark_results_a.json", {"dataset": {"name": "a", "n": 1}}, mtime_offset=-300)
    _write_run(tmp_path, "benchmark_results_b.json", {"dataset": {"name": "b", "n": 1}}, mtime_offset=-150)
    _write_run(tmp_path, "benchmark_results_c.json", {"dataset": {"name": "c", "n": 1}}, mtime_offset=0)

    result = list_benchmark_runs(directory=str(tmp_path))
    names_in_order = [r["dataset"]["name"] for r in result["runs"]]
    assert names_in_order == ["c", "b", "a"]


def test_malformed_file_among_valid_ones_is_skipped_not_fatal(tmp_path):
    _write_run(tmp_path, "benchmark_results_good1.json", {"dataset": {"name": "good1", "n": 1}}, mtime_offset=-10)
    bad_path = tmp_path / "benchmark_results_broken.json"
    bad_path.write_text('{"config": {"a": 1}, "dataset": ', encoding="utf-8")  # truncated JSON
    _write_run(tmp_path, "benchmark_results_good2.json", {"dataset": {"name": "good2", "n": 1}}, mtime_offset=0)

    result = list_benchmark_runs(directory=str(tmp_path))

    good_names = {r["dataset"]["name"] for r in result["runs"]}
    assert good_names == {"good1", "good2"}
    assert len(result["runs"]) == 2

    assert len(result["errors"]) == 1
    assert result["errors"][0]["filename"] == "benchmark_results_broken.json"
    assert "error" in result["errors"][0] and result["errors"][0]["error"]


def test_empty_directory_returns_no_runs_and_no_errors(tmp_path):
    result = list_benchmark_runs(directory=str(tmp_path))
    assert result == {"runs": [], "errors": []}


def test_missing_directory_returns_no_runs_and_no_errors(tmp_path):
    missing = tmp_path / "nonexistent_subdir"
    assert not missing.exists()
    result = list_benchmark_runs(directory=str(missing))
    assert result == {"runs": [], "errors": []}


def test_unexpected_shape_file_matching_pattern_is_flagged_as_error_not_crash(tmp_path):
    """Trip wire from the loader's defensive-handling side: a file whose
    name matches benchmark_results_*.json but whose top-level JSON isn't
    an object (e.g. a bare list or scalar, as could happen if a future
    benchmark script's output format drifts) must be reported in
    `errors`, not raise, and must not appear among `runs`."""
    # A list at the top level -- the trip wire this module explicitly guards.
    (tmp_path / "benchmark_results_wrong_shape_list.json").write_text(
        json.dumps([{"config": {}, "dataset": {}}]), encoding="utf-8"
    )
    # A bare scalar -- an even more drifted shape.
    (tmp_path / "benchmark_results_wrong_shape_scalar.json").write_text(
        json.dumps(42), encoding="utf-8"
    )
    _write_run(tmp_path, "benchmark_results_normal.json", {"dataset": {"name": "normal", "n": 1}})

    result = list_benchmark_runs(directory=str(tmp_path))

    assert len(result["runs"]) == 1
    assert result["runs"][0]["dataset"]["name"] == "normal"

    error_filenames = {e["filename"] for e in result["errors"]}
    assert error_filenames == {
        "benchmark_results_wrong_shape_list.json",
        "benchmark_results_wrong_shape_scalar.json",
    }
    for e in result["errors"]:
        assert "expected a JSON object at the top level" in e["error"]
