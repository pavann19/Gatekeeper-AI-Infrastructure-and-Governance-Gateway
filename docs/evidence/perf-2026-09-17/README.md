# Historical throughput measurements

These are historical single-machine throughput measurements from the 2026-09-17 collection, retained unchanged from source commit `bfc185d76f34820cd373e5c0c34d0f29d376b85c`.

| File | Concurrency | Measured requests | Throughput (requests/s) | Error rate |
|---|---:|---:|---:|---:|
| [acc15aab67d7-4.json](acc15aab67d7-4.json) | 4 | 240 | 16.0 | 0.0 |
| [b59b57417e41-1.json](b59b57417e41-1.json) | 1 | 441 | 7.35 | 0.0 |
| [b59b57417e41-16.json](b59b57417e41-16.json) | 16 | 962 | 16.033 | 0.0 |
| [b59b57417e41-4.json](b59b57417e41-4.json) | 4 | 943 | 15.717 | 0.0 |
| [e55452550ae2-1.json](e55452550ae2-1.json) | 1 | 332 | 11.067 | 0.0 |
| [e55452550ae2-16.json](e55452550ae2-16.json) | 16 | 309 | 10.3 | 0.0 |
| [e55452550ae2-4.json](e55452550ae2-4.json) | 4 | 409 | 13.633 | 0.0 |
| [run_20260917_001922/e55452550ae2-1.json](run_20260917_001922/e55452550ae2-1.json) | 1 | 340 | 11.333 | 0.0 |
| [run_20260917_001922/e55452550ae2-16.json](run_20260917_001922/e55452550ae2-16.json) | 16 | 367 | 12.233 | 0.0 |
| [run_20260917_001922/e55452550ae2-4.json](run_20260917_001922/e55452550ae2-4.json) | 4 | 423 | 14.1 | 0.0 |
| [run_20260917_002918/e55452550ae2-1.json](run_20260917_002918/e55452550ae2-1.json) | 1 | 283 | 9.433 | 0.0 |
| [run_20260917_002918/e55452550ae2-16.json](run_20260917_002918/e55452550ae2-16.json) | 16 | 307 | 10.233 | 0.0 |
| [run_20260917_002918/e55452550ae2-4.json](run_20260917_002918/e55452550ae2-4.json) | 4 | 432 | 14.4 | 0.0 |
| [run_20260917_004034/judge_raw_response.json](run_20260917_004034/judge_raw_response.json) | not recorded | not recorded | not recorded | not recorded |
| [run_20260917_004655/e55452550ae2-1.json](run_20260917_004655/e55452550ae2-1.json) | 1 | 332 | 11.067 | 0.0 |
| [run_20260917_004655/e55452550ae2-16.json](run_20260917_004655/e55452550ae2-16.json) | 16 | 309 | 10.3 | 0.0 |
| [run_20260917_004655/e55452550ae2-4.json](run_20260917_004655/e55452550ae2-4.json) | 4 | 409 | 13.633 | 0.0 |

The baseline `e55452550ae2` files record `judge_label` as `disabled_unreachable`; the later `b59b57417e41` and `acc15aab67d7` files record `unspecified`. These are different recorded judge settings. Single-request throughput was lower in the later run: `b59b57417e41-1.json` records 7.35 requests/s, compared with 11.067 requests/s in the top-level `e55452550ae2-1.json` baseline.

These receipts are not a current published throughput headline and are not accuracy evidence. See the [evaluation notes](../../EVALUATION.md).
