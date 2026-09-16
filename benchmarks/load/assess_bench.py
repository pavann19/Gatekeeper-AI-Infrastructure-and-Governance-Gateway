"""
Closed-loop concurrency benchmark for POST /api/v1/assess.

Fixes every problem the previous benchmarks/run_load_test.py had:
  - fixed seed / fixed workload (workload_v1.jsonl, hash-verified before run)
  - closed-loop concurrency (N workers, each immediately re-issuing on
    completion) at fixed levels, not an open-loop RPS ramp
  - a warm-up period, discarded from the measured window
  - sends Authorization: Bearer <key> -- an anonymous caller is capped at
    RATE_LIMIT_ANONYMOUS_RPM (20 RPM vs 120 authenticated; see
    core/config.py), which would measure the rate limiter, not the pipeline
  - one real f-string newline in every print, not a literal "\\n"

    PYTHONPATH=. python benchmarks/load/assess_bench.py \\
        --api-key "$GATEKEEPER_BENCH_API_KEY" \\
        --concurrency 1,4,16 --duration 60 --warmup 10 \\
        --judge-label enabled

Requires a Gatekeeper API already running (same checkout as this script --
see --repo-root) and reachable at --base-url. Provision a benchmark key
first:

    PYTHONPATH=. python scripts/manage_api_keys.py issue \\
        --capability INTERNAL --tenant benchmark --label "load-test"

Writes one JSON file per concurrency level to
  _evidence/perf/<git-sha>-<concurrency>.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKLOAD = os.path.join(HERE, "workload_v1.jsonl")
PERF_DIR = os.path.join(REPO, "_evidence", "perf")


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

def load_workload() -> list[dict]:
    sha_file = WORKLOAD + ".sha256"
    if not os.path.exists(WORKLOAD) or not os.path.exists(sha_file):
        raise SystemExit(
            f"missing {WORKLOAD} or its .sha256 sidecar -- run "
            "benchmarks/load/build_workload.py first."
        )
    expected = open(sha_file, encoding="utf-8").read().split()[0]
    actual = hashlib.sha256(open(WORKLOAD, "rb").read()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"workload_v1.jsonl does not match its committed hash "
            f"(expected {expected}, got {actual}) -- the fixed workload was "
            f"modified. Re-run build_workload.py deliberately if that was "
            f"intended, and commit the new .sha256 alongside it."
        )
    rows = [json.loads(line) for line in open(WORKLOAD, encoding="utf-8")]
    if not rows:
        raise SystemExit("workload_v1.jsonl is empty")
    return rows


# ---------------------------------------------------------------------------
# Metadata: hardware, config, model pins -- so a later run is comparable
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        )
        return bool(out.strip())
    except Exception:
        return True  # unknown -> assume dirty, don't claim a clean baseline


def collect_environment_metadata() -> dict:
    """
    Everything needed to tell two runs apart later: hardware, thread
    settings, the concurrency knob, and which exact model revisions were
    pinned. Reads core/config.py and models/detector_manifest.json directly
    from THIS checkout -- only valid when the server under test was started
    from the same one (true for a local baseline run; a remote target needs
    this re-derived from that host instead).
    """
    import sys
    sys.path.insert(0, REPO)
    from core.config import settings  # noqa: E402

    try:
        import torch
        torch_threads = torch.get_num_threads()
        torch_interop_threads = torch.get_num_interop_threads()
    except Exception as e:  # noqa: BLE001
        torch_threads = torch_interop_threads = f"unavailable: {e}"

    manifest_path = os.path.join(REPO, "models", "detector_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}
    model_revisions = {k: v.get("revision") for k, v in manifest.items()}

    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
        },
        "concurrency_config": {
            "ASSESS_MAX_CONCURRENCY": settings.ASSESS_MAX_CONCURRENCY,
            "ASSESS_TIMEOUT_SECONDS": settings.ASSESS_TIMEOUT_SECONDS,
        },
        "torch_threads": {
            "intra_op": torch_threads,
            "inter_op": torch_interop_threads,
            "OMP_NUM_THREADS_env": os.environ.get("OMP_NUM_THREADS"),
            "note": "no explicit torch.set_num_threads() call exists in this "
                    "codebase as of this run -- these are torch's own defaults, "
                    "not a deliberate setting. See docs/perf/01-baseline-analysis.md.",
        },
        "model_revisions": model_revisions,
    }


async def probe_judge_backend(ollama_api_url: str, timeout: float = 2.0) -> bool:
    base = "/".join(ollama_api_url.split("/")[:-1])
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base}/tags")
            return r.status_code == 200
    except Exception:
        return False


async def scrape_metrics(client: httpx.AsyncClient, base_url: str, headers: dict) -> str | None:
    try:
        r = await client.get(f"{base_url}/metrics", headers=headers, timeout=5.0)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def _parse_prom_gauge(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
    return None


def _parse_prom_counter_total(text: str, name: str) -> float:
    """Sums every label combination of a counter (judge_invocations_total
    has none, but this stays correct if that ever changes)."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
    return total


# ---------------------------------------------------------------------------
# Load generation -- closed loop
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    latencies_ms: list = field(default_factory=list)   # measured window only
    statuses: list = field(default_factory=list)
    warmup_count: int = 0
    measured_count: int = 0


async def _worker(worker_id, client, base_url, headers, workload, deadline, warmup_until, result: RunResult, lock):
    i = worker_id
    n = len(workload)
    while time.monotonic() < deadline:
        row = workload[i % n]
        i += n
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{base_url}/api/v1/assess",
                json={"prompt": row["text"]},
                headers=headers,
                timeout=30.0,
            )
            status = r.status_code
        except Exception:
            status = 0
        dt_ms = (time.perf_counter() - t0) * 1000.0
        now = time.monotonic()
        async with lock:
            if now < warmup_until:
                result.warmup_count += 1
            else:
                result.measured_count += 1
                result.latencies_ms.append(dt_ms)
                result.statuses.append(status)


def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[k]


async def run_one_concurrency(
    concurrency: int, base_url: str, headers: dict, workload: list[dict],
    duration: float, warmup: float, metrics_interval: float,
) -> dict:
    result = RunResult()
    lock = asyncio.Lock()
    start = time.monotonic()
    warmup_until = start + warmup
    deadline = start + warmup + duration

    limits = httpx.Limits(max_connections=concurrency + 5, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        in_flight_peak = 0.0
        judge_before = judge_after = None

        async def poll_metrics():
            nonlocal in_flight_peak, judge_before, judge_after
            while time.monotonic() < deadline:
                text = await scrape_metrics(client, base_url, headers)
                if text:
                    v = _parse_prom_gauge(text, "gatekeeper_assessments_in_flight")
                    if v is not None:
                        in_flight_peak = max(in_flight_peak, v)
                    j = _parse_prom_counter_total(text, "gatekeeper_judge_invocations_total")
                    if time.monotonic() < warmup_until:
                        judge_before = j
                    judge_after = j
                await asyncio.sleep(metrics_interval)

        poller = asyncio.create_task(poll_metrics())
        workers = [
            asyncio.create_task(
                _worker(w, client, base_url, headers, workload, deadline, warmup_until, result, lock)
            )
            for w in range(concurrency)
        ]
        await asyncio.gather(*workers)
        await poller

    lat = sorted(result.latencies_ms)
    n_2xx = sum(1 for s in result.statuses if 200 <= s < 300)
    n_total = len(result.statuses)
    error_rate = (n_total - n_2xx) / n_total if n_total else float("nan")
    status_counts: dict = {}
    for s in result.statuses:
        key = str(s)
        status_counts[key] = status_counts.get(key, 0) + 1

    judge_delta = None
    if judge_before is not None and judge_after is not None:
        judge_delta = max(0.0, judge_after - judge_before)

    return {
        "concurrency": concurrency,
        "warmup_seconds": warmup,
        "duration_seconds": duration,
        "requests": {
            "warmup_discarded": result.warmup_count,
            "measured_total": n_total,
            "measured_2xx": n_2xx,
        },
        "throughput_rps": round(n_total / duration, 3) if duration else None,
        "error_rate": round(error_rate, 4) if n_total else None,
        "status_counts": status_counts,
        "latency_ms": {
            "p50": round(_percentile(lat, 0.50), 2) if lat else None,
            "p95": round(_percentile(lat, 0.95), 2) if lat else None,
            "p99": round(_percentile(lat, 0.99), 2) if lat else None,
            "min": round(lat[0], 2) if lat else None,
            "max": round(lat[-1], 2) if lat else None,
        },
        "assessments_in_flight_peak": in_flight_peak,
        "judge_invocations_during_measured_window_approx": judge_delta,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    workload = load_workload()
    workload_sha = open(WORKLOAD + ".sha256", encoding="utf-8").read().split()[0]

    headers = {}
    api_key = args.api_key or os.environ.get("GATEKEEPER_BENCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif not args.allow_anonymous:
        raise SystemExit(
            "no API key given (--api-key or GATEKEEPER_BENCH_API_KEY). An "
            "anonymous caller is capped at RATE_LIMIT_ANONYMOUS_RPM (20/min) "
            "vs 120/min authenticated -- see core/config.py -- so an "
            "unauthenticated run measures the rate limiter, not the "
            "pipeline. Issue a key with scripts/manage_api_keys.py, or pass "
            "--allow-anonymous to do this deliberately anyway."
        )

    from core.config import settings  # noqa: E402  (sys.path set in collect_environment_metadata)
    judge_reachable = await probe_judge_backend(settings.OLLAMA_API_URL)

    env_meta = collect_environment_metadata()
    print(f"git_sha={env_meta['git_sha'][:12]}  dirty={env_meta['git_dirty']}  "
          f"cpu_count={env_meta['hardware']['cpu_count']}  "
          f"ASSESS_MAX_CONCURRENCY={env_meta['concurrency_config']['ASSESS_MAX_CONCURRENCY']}\n"
          f"judge backend reachable: {judge_reachable}  "
          f"(--judge-label={args.judge_label!r} is the run's own claim; this probe is the check)")

    os.makedirs(PERF_DIR, exist_ok=True)
    levels = [int(x) for x in args.concurrency.split(",")]

    for level in levels:
        print(f"\n--- concurrency={level}  warmup={args.warmup}s  duration={args.duration}s ---")
        result = await run_one_concurrency(
            level, args.base_url, headers, workload,
            duration=args.duration, warmup=args.warmup,
            metrics_interval=args.metrics_interval,
        )
        payload = {
            "benchmark": "assess_bench",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workload_file": os.path.relpath(WORKLOAD, REPO),
            "workload_sha256": workload_sha,
            "judge_label": args.judge_label,
            "judge_backend_reachable_at_start": judge_reachable,
            "environment": env_meta,
            "result": result,
        }
        print(f"  throughput={result['throughput_rps']} rps  "
              f"error_rate={result['error_rate']}  "
              f"p50={result['latency_ms']['p50']}ms  "
              f"p95={result['latency_ms']['p95']}ms  "
              f"p99={result['latency_ms']['p99']}ms  "
              f"in_flight_peak={result['assessments_in_flight_peak']}")

        out_path = os.path.join(
            PERF_DIR, f"{env_meta['git_sha'][:12]}-{level}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  saved: {os.path.relpath(out_path, REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--allow-anonymous", action="store_true",
                    help="run without an API key anyway (measures the rate limiter too)")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--duration", type=float, default=60.0, help="measured seconds per level")
    ap.add_argument("--warmup", type=float, default=10.0, help="discarded seconds per level")
    ap.add_argument("--metrics-interval", type=float, default=1.0)
    ap.add_argument("--judge-label", default="unspecified",
                    help="free-text label for this run's judge state, e.g. "
                         "'enabled' or 'disabled_unreachable' -- verified "
                         "against a live probe, not trusted blindly")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
