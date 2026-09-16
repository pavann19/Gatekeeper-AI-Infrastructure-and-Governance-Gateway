# G3 reproducibility: one command that proves a fresh clone actually works.
#
#   git clone <repo> && cd <repo>
#   cp .env.example .env
#   docker compose up -d
#   make verify
#
# `verify` runs the unit tests (pass without any model weights present --
# real-model tests skip cleanly via importorskip, see tests/test_model_pinning.py
# and tests/test_fault_injection.py), then downloads and warms every pinned
# detector via a real server start, then a 20-prompt smoke eval, then a 30s
# closed-loop benchmark. RAM-gated throughout (scripts/verify.sh) -- a
# resource-constrained clone gets a clear SKIP + reason instead of a crash.

.PHONY: verify test smoke bench

verify:
	bash scripts/verify.sh

test:
	rm -f audit.db audit.db-wal audit.db-shm
	PYTHONPATH=. python -m pytest tests/ -q
	rm -f audit.db audit.db-wal audit.db-shm

smoke:
	PYTHONPATH=. python scripts/smoke_eval.py --n 20

bench:
	PYTHONPATH=. python benchmarks/load/assess_bench.py --allow-anonymous --concurrency 1,4,16 --duration 30 --warmup 5
