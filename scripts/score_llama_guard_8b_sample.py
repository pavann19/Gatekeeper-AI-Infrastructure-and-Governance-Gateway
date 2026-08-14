"""
Scores Llama Guard 3 8B (quantized, via Ollama) on the SAME deterministic
554-row stratified sample used for the 1B run
(scripts/eval_llama_guard_sample.py), so the two are directly comparable --
same rows, same class composition, same seed.

WHY A SAMPLE INSTEAD OF THE FULL SUITE: full-suite scoring at ~11s/row is
~21 hours, which does not fit a single overnight window and, worse, an early
representative-mix check (807/6933 rows) already showed 8B trending WEAKER
on harmful_content recall than 1B (58.3% vs 73.6%) -- continuing the full run
on the strength of a hope it reverses is exactly the "measure the wrong thing
because it was easy" mistake eval_llama_guard_sample.py's own docstring warns
against. A complete, comparable sample in ~1.5h beats an incomplete,
non-comparable full run in ~21h.

REUSES WORK ALREADY DONE: rows already scored during the (abandoned)
full-suite attempt are read from _evidence/detector_scores/llama_guard_3_8b.jsonl
first and copied into this sample's own cache file, so nothing already paid
for is re-scored.

Usage:
    python -m scripts.score_llama_guard_8b_sample
"""
import json
import os
import sys

sys.path.insert(0, ".")

from scripts.eval_llama_guard_sample import build_sample, load_suite
from scripts.score_llama_guard_ollama import (
    OLLAMA_BASE_URL,
    ollama_reachable,
    score_one,
)

SAMPLE_CACHE = os.path.join("_evidence", "detector_scores", "llama_guard_3_8b.sample.jsonl")
FULL_RUN_CACHE = os.path.join("_evidence", "detector_scores", "llama_guard_3_8b.jsonl")
PROGRESS_FILE = os.path.join("_evidence", "llama_guard_8b_sample_progress.json")

PER_OTHER_CLASS = 50
BENIGN_N = 200


def load_scores(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out[r["id"]] = r["score"]
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main():
    import time

    rows = load_suite()
    sample, composition = build_sample(rows, PER_OTHER_CLASS, BENIGN_N)
    sample_ids = {r["id"] for r in sample}
    print(f"Sample: {len(sample)} rows, composition {composition}")

    already_in_sample_cache = load_scores(SAMPLE_CACHE)
    from_full_run = load_scores(FULL_RUN_CACHE)

    # Seed the sample cache with anything already scored during the abandoned
    # full-suite run, so it is never redone.
    carried_over = 0
    with open(SAMPLE_CACHE, "a", encoding="utf-8") as f:
        for sid in sample_ids:
            if sid in already_in_sample_cache:
                continue
            if sid in from_full_run:
                f.write(json.dumps({"id": sid, "score": from_full_run[sid]}) + "\n")
                already_in_sample_cache[sid] = from_full_run[sid]
                carried_over += 1
        f.flush()
    if carried_over:
        print(f"Carried over {carried_over} rows already scored during the "
              f"full-suite attempt.")

    todo = [r for r in sample if r["id"] not in already_in_sample_cache]
    print(f"Already have {len(sample) - len(todo)}/{len(sample)}. "
          f"Remaining: {len(todo)}.")
    sys.stdout.flush()

    if not todo:
        print("Nothing to do -- sample fully scored.")
        return

    if not ollama_reachable():
        raise SystemExit("Ollama is not reachable at localhost:11434. Start it first.")

    started_at = time.time()
    consecutive_failures = 0

    with open(SAMPLE_CACHE, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo):
            last_error = None
            for attempt in range(1, 4):
                try:
                    score = score_one(row["text"])
                    out.write(json.dumps({"id": row["id"], "score": score}) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    consecutive_failures = 0
                    break
                except Exception as e:
                    last_error = f"{type(e).__name__}: {str(e)[:150]}"
                    consecutive_failures += 1
                    print(f"  [{i+1}/{len(todo)}] attempt {attempt}/3 failed: {last_error}")
                    sys.stdout.flush()
                    if attempt < 3:
                        time.sleep(5)
            else:
                print(f"  [{i+1}/{len(todo)}] GIVING UP on row {row['id']}: {last_error}")
                sys.stdout.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                elapsed_m = (time.time() - started_at) / 60
                done_total = len(sample) - len(todo) + i + 1
                rate = (i + 1) / (time.time() - started_at) if time.time() > started_at else 0
                remaining_m = (len(todo) - i - 1) / rate / 60 if rate > 0 else None
                with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
                    json.dump({
                        "done": done_total, "total": len(sample),
                        "elapsed_this_run_m": round(elapsed_m, 1),
                        "eta_remaining_m": round(remaining_m, 1) if remaining_m else None,
                        "last_error": last_error,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, pf, indent=2)
                print(f"  progress: {done_total}/{len(sample)} total "
                      f"({i+1}/{len(todo)} this run) | {elapsed_m:.1f}m elapsed")
                sys.stdout.flush()

            if consecutive_failures >= 15:
                print("Ollama appears down. Stopping cleanly; rerun this script "
                      "once reachable again -- it resumes.")
                raise SystemExit(1)

    print(f"\nDone. {len(sample)}/{len(sample)} rows scored -> {SAMPLE_CACHE}")


if __name__ == "__main__":
    main()
