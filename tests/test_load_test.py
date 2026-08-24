"""
Tests for scripts/load_test.py's aggregation logic (percentiles, throughput,
status histogram) -- the actual HTTP layer (`_one_request`) is mocked here
so these run without a live server; the tool itself was additionally run
against a real running instance (see docs/ROADMAP_V2.md's Phase 8 entry)
to produce real evidence, which is what a unit test alone can't prove.
"""
from unittest.mock import patch

from scripts.load_test import run_load_test


def _fake_results(elapsed_list, status=200):
    return [{"status": status, "elapsed": e, "error": None} for e in elapsed_list]


def test_throughput_and_latency_percentiles_computed_correctly():
    fake = _fake_results([0.01, 0.02, 0.03, 0.04, 0.10])
    with patch("scripts.load_test._one_request", side_effect=fake):
        result = run_load_test("http://x", "/e", concurrency=1, total_requests=5)

    assert result["total_requests"] == 5
    assert result["status_counts"] == {"200": 5}
    assert result["latency_ms"]["min"] == 10.0
    assert result["latency_ms"]["max"] == 100.0


def test_mixed_status_codes_are_all_counted():
    responses = [
        {"status": 200, "elapsed": 0.01, "error": None},
        {"status": 200, "elapsed": 0.01, "error": None},
        {"status": 429, "elapsed": 0.01, "error": None},
        {"status": 500, "elapsed": 0.01, "error": None},
    ]
    with patch("scripts.load_test._one_request", side_effect=responses):
        result = run_load_test("http://x", "/e", concurrency=1, total_requests=4)

    assert result["status_counts"] == {"200": 2, "429": 1, "500": 1}


def test_connection_errors_are_counted_separately_from_http_statuses():
    responses = [
        {"status": None, "elapsed": 0.01, "error": "Connection refused"},
        {"status": 200, "elapsed": 0.01, "error": None},
    ]
    with patch("scripts.load_test._one_request", side_effect=responses):
        result = run_load_test("http://x", "/e", concurrency=1, total_requests=2)

    assert result["status_counts"] == {"connection_error": 1, "200": 1}


def test_throughput_is_requests_over_real_wall_clock_elapsed():
    fake = _fake_results([0.001] * 10)
    with patch("scripts.load_test._one_request", side_effect=fake):
        result = run_load_test("http://x", "/e", concurrency=5, total_requests=10)

    assert result["throughput_rps"] > 0
    assert result["total_elapsed_s"] > 0


def test_zero_requests_does_not_crash():
    with patch("scripts.load_test._one_request", side_effect=[]):
        result = run_load_test("http://x", "/e", concurrency=1, total_requests=0)

    assert result["status_counts"] == {}
    assert result["latency_ms"]["min"] is None
    # 0 completed requests over any positive elapsed time is a defined 0
    # rps, not an undefined value -- None is reserved for "elapsed was
    # somehow non-positive", which real wall-clock time never is.
    assert result["throughput_rps"] == 0.0


def test_url_is_correctly_joined_from_base_and_endpoint():
    captured = {}

    def fake_one_request(url, headers, method, json_body, timeout):
        captured["url"] = url
        return {"status": 200, "elapsed": 0.001, "error": None}

    with patch("scripts.load_test._one_request", side_effect=fake_one_request):
        run_load_test("http://x:8000/", "/api/v1/activity", concurrency=1, total_requests=1)

    assert captured["url"] == "http://x:8000/api/v1/activity"


def test_api_key_becomes_a_bearer_header():
    captured = {}

    def fake_one_request(url, headers, method, json_body, timeout):
        captured["headers"] = headers
        return {"status": 200, "elapsed": 0.001, "error": None}

    with patch("scripts.load_test._one_request", side_effect=fake_one_request):
        run_load_test("http://x", "/e", concurrency=1, total_requests=1, api_key="gk_test123")

    assert captured["headers"] == {"Authorization": "Bearer gk_test123"}
