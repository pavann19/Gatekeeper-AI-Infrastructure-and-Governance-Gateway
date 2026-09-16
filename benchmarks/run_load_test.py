"""
RETIRED. This tool had no fixed seed, no warm-up period, no auth header (so
it measured the anonymous rate limiter, not the pipeline), three hardcoded
prompts, and printed a literal "\\n" instead of a newline.

Replaced by benchmarks/load/assess_bench.py, a closed-loop concurrency
benchmark against a fixed, hash-verified workload
(benchmarks/load/workload_v1.jsonl). See that file's docstring, and
docs/perf/01-baseline-analysis.md for the first baseline run.

    PYTHONPATH=. python benchmarks/load/assess_bench.py --api-key "$KEY"
"""
raise SystemExit(
    "benchmarks/run_load_test.py is retired -- use "
    "benchmarks/load/assess_bench.py (see its docstring)."
)
