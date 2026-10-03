# Public Benchmark Summary

This repository publishes benchmark summaries and methodology, not raw run
artifacts. Raw logs, CSV rows, JSONL streams, generated flamegraphs, and
machine-specific `_evidence/` captures are local evidence files and are ignored
by Git.

## Accuracy Snapshot

End-to-end benchmark against `deepset/prompt-injections`, 546 prompts, domain
guardrail off, `llama-guard3` judge configured:

| mode | accuracy | precision | recall | FPR | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold cache | 80.8% | 83.1% | 60.6% | 7.3% | 1,028 ms | 1,810 ms | 2,681 ms |
| warm cache | 80.8% | 83.1% | 60.6% | 7.3% | 56.5 ms | 123.9 ms | 253.6 ms |

Confusion matrix for both cold and warm cache: TP 123, FP 25, TN 318, FN 80.
The warm-cache speedup was 16.6x on this workload.

## Authenticated Load Snapshot

Closed-loop `/api/v1/assess` benchmark on the fixed workload
`benchmarks/load/workload_v1.jsonl`, SHA-256
`55e725aefc954fe9b936c1e8d4b9aec7c71e5d7e24e76c5fa6da6a219c9be98a`, measured
on the local Windows 11, 12-core development machine at commit `007c8b5`.

| concurrency | measured requests | 2xx | throughput | p50 | p95 | p99 | error rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 459 | 459 | 7.65 rps | 103.5 ms | 367.1 ms | 376.9 ms | 0.0% |
| 4 | 961 | 961 | 16.02 rps | 187.2 ms | 500.0 ms | 622.1 ms | 0.0% |
| 16 | 970 | 970 | 16.17 rps | 960.2 ms | 1,335.5 ms | 2,449.0 ms | 0.0% |

The judge backend was unreachable at the start of this load run, so judge-path
latency is not claimed from these numbers.

## Evidence Boundary

The numbers above are public summaries extracted from local benchmark outputs.
The raw evidence files remain local because they include noisy operational logs,
row-level CSVs, machine-specific traces, generated flamegraphs, and transient
diagnostic captures that are useful for audit but not appropriate as public
repository content.
