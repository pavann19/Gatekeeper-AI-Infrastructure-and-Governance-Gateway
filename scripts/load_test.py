"""
Real load test against a running Gatekeeper instance (Phase 8: "Load
testing"). Opens real HTTP connections to a real running server with real
concurrent threads and reports real latency percentiles, throughput, and
status-code distribution -- not a simulated or mocked estimate.

Usage:
    python -m scripts.load_test --url http://127.0.0.1:8000 \
        --endpoint /api/v1/activity --concurrency 20 --requests 500 \
        --api-key gk_... [--out _evidence/load_test_results.json]

Defaults to a read-only endpoint (the activity feed) so running this
against a real deployment by default cannot mutate state or trigger the
detection pipeline's real (and deliberately rate-limited) cost. Pointing
it at a write/assess endpoint is opt-in via --method/--body.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time

import requests


def _one_request(url, headers, method, json_body, timeout):
    start = time.perf_counter()
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        else:
            response = requests.post(url, headers=headers, json=json_body, timeout=timeout)
        elapsed = time.perf_counter() - start
        return {"status": response.status_code, "elapsed": elapsed, "error": None}
    except requests.RequestException as e:
        elapsed = time.perf_counter() - start
        return {"status": None, "elapsed": elapsed, "error": str(e)}


def _percentile(sorted_values, p):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * p))
    return sorted_values[idx]


def run_load_test(base_url, endpoint, concurrency, total_requests,
                  api_key=None, method="GET", json_body=None, timeout=30):
    """
    Fires `total_requests` real HTTP requests at `base_url + endpoint`
    across `concurrency` real threads, and returns a summary dict: real
    throughput (requests actually completed / real wall-clock elapsed),
    a real status-code histogram, and real latency percentiles in ms.

    Pure aggregation logic, separated from `main()`'s CLI/file-I/O
    concerns so it's directly unit-testable with `_one_request` mocked
    (see tests/test_load_test.py) without needing a live server for
    every test run.
    """
    url = base_url.rstrip("/") + endpoint
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    results = []
    start_all = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_one_request, url, headers, method, json_body, timeout)
            for _ in range(total_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    total_elapsed = time.perf_counter() - start_all

    latencies_ms = sorted(r["elapsed"] * 1000 for r in results)
    status_counts: dict = {}
    for r in results:
        key = str(r["status"]) if r["status"] is not None else "connection_error"
        status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "endpoint": endpoint,
        "method": method,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_elapsed_s": round(total_elapsed, 3),
        "throughput_rps": round(total_requests / total_elapsed, 2) if total_elapsed > 0 else None,
        "status_counts": status_counts,
        "latency_ms": {
            "min": round(latencies_ms[0], 2) if latencies_ms else None,
            "p50": round(_percentile(latencies_ms, 0.50), 2) if latencies_ms else None,
            "p95": round(_percentile(latencies_ms, 0.95), 2) if latencies_ms else None,
            "p99": round(_percentile(latencies_ms, 0.99), 2) if latencies_ms else None,
            "max": round(latencies_ms[-1], 2) if latencies_ms else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/api/v1/activity")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--method", default="GET", choices=["GET", "POST"])
    parser.add_argument("--body", default=None, help="JSON string, for --method POST")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="Path to save results as JSON")
    args = parser.parse_args()

    json_body = json.loads(args.body) if args.body else None
    result = run_load_test(
        args.url, args.endpoint, args.concurrency, args.requests,
        api_key=args.api_key, method=args.method, json_body=json_body, timeout=args.timeout,
    )

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
