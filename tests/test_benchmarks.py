"""
Tests for core/benchmarks.py -- reads local benchmark result files. Uses
tmp_path for synthetic cases (missing dir, corrupt file, sorting, and
realistic public-summary-shaped content) so the public repo no longer depends
on tracked raw `_evidence/` artifacts.
"""
import json
import os

from core.benchmarks import list_benchmark_runs


def test_missing_directory_is_empty_not_an_error(tmp_path):
    result = list_benchmark_runs(directory=str(tmp_path / "does_not_exist"))
    assert result == {"runs": [], "errors": []}


def test_empty_directory_is_empty(tmp_path):
    result = list_benchmark_runs(directory=str(tmp_path))
    assert result == {"runs": [], "errors": []}


def test_lists_a_real_benchmark_file(tmp_path):
    content = {"config": {"a": 1}, "dataset": {"name": "x", "n": 10}, "cold": {}, "warm": {}}
    (tmp_path / "benchmark_results_run1.json").write_text(json.dumps(content), encoding="utf-8")
    result = list_benchmark_runs(directory=str(tmp_path))
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert run["dataset"] == {"name": "x", "n": 10}
    assert run["_filename"] == "benchmark_results_run1.json"
    assert "_mtime" in run


def test_only_matches_the_benchmark_results_naming_pattern(tmp_path):
    (tmp_path / "benchmark_results_a.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    (tmp_path / "calibration_report.json").write_text(json.dumps({"y": 2}), encoding="utf-8")
    (tmp_path / "detector_comparison.json").write_text(json.dumps({"z": 3}), encoding="utf-8")
    result = list_benchmark_runs(directory=str(tmp_path))
    assert len(result["runs"]) == 1
    assert result["runs"][0]["_filename"] == "benchmark_results_a.json"


def test_newest_file_first(tmp_path):
    old = tmp_path / "benchmark_results_old.json"
    new = tmp_path / "benchmark_results_new.json"
    old.write_text(json.dumps({"which": "old"}), encoding="utf-8")
    new.write_text(json.dumps({"which": "new"}), encoding="utf-8")
    older_time = os.path.getmtime(old) - 100
    os.utime(old, (older_time, older_time))
    result = list_benchmark_runs(directory=str(tmp_path))
    assert [r["which"] for r in result["runs"]] == ["new", "old"]


def test_corrupt_file_reported_as_error_not_silently_dropped(tmp_path):
    (tmp_path / "benchmark_results_good.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (tmp_path / "benchmark_results_bad.json").write_text("{ not valid json", encoding="utf-8")
    result = list_benchmark_runs(directory=str(tmp_path))
    assert len(result["runs"]) == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0]["filename"] == "benchmark_results_bad.json"


def test_non_object_json_is_an_error(tmp_path):
    (tmp_path / "benchmark_results_list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    result = list_benchmark_runs(directory=str(tmp_path))
    assert result["runs"] == []
    assert len(result["errors"]) == 1


def test_realistic_public_summary_shape_parses(tmp_path):
    """Trip wire for the shape the UI assumes without committing raw evidence."""
    content = {
        "config": {"domain_guardrail_mode": "off"},
        "dataset": {"name": "deepset/prompt-injections", "n": 546},
        "cold": {"metrics_operational": {"accuracy": 0.807}},
        "warm": {"metrics_operational": {"accuracy": 0.807}},
    }
    (tmp_path / "benchmark_results_public_shape.json").write_text(
        json.dumps(content), encoding="utf-8"
    )
    result = list_benchmark_runs(directory=str(tmp_path))
    assert result["errors"] == []
    assert result["runs"][0]["dataset"]["n"] == 546
    assert "cold" in result["runs"][0]
    assert "warm" in result["runs"][0]
