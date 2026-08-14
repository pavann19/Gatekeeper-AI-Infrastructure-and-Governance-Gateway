"""
Scores the full evaluation suite against Llama Guard 3 8B, served quantized
(Q4_K_M, 4.92GB) via Ollama -- the only way this model runs on a 12GB laptop
at all. The in-process HF detector (core/detectors.py's llama_guard_3_8b)
requires min_free_gb=18.0 and refuses to load here by design; this script is
a separate, Ollama-backed path to the SAME model family, for offline scoring
only. It does not touch the live judge-arbitration code path.

WHY A SEPARATE SCRIPT rather than adding Ollama support to
scripts/compare_detectors.py's Detector interface: that interface assumes an
in-process `score_batch()` call with predictable latency (milliseconds).
Ollama's 8B model measured ~7.5s/prompt earlier this session -- three orders
of magnitude slower -- and needs its own resumability, health-check, and
progress-reporting story that would be dead weight on every other detector.

OUTPUT FORMAT matches scripts/compare_detectors.py's cache exactly
({"id": ..., "score": ...} per line in _evidence/detector_scores/), so once
this finishes, `python -m scripts.compare_detectors --only llama_guard_3_8b`
(after registering the cache) or a direct read of the .jsonl slots straight
into the existing evaluate_detector() pipeline with no format conversion.

SCORE MEANING: Llama Guard has no probability output, only a binary
safe/unsafe verdict (core/semantic_judge.py's _judge_via_llama_guard
documents this same constraint for the live judge path). Score is therefore
1.0 (unsafe) or 0.0 (safe) -- a step function, not a smooth probability. AUC
computed from step-function scores is a legitimate but coarser statistic than
a real probability would give; this is a property of the model, not a bug in
this script.

RESUMABILITY is the entire point of an unattended overnight run. Every
successfully scored row is appended to the output file immediately, and a
restart (of Ollama, or of this script) reads existing output first and skips
any id already present. Killing this process and rerunning it costs at most
the one in-flight row, never previously-completed work.

Usage:
    python -m scripts.score_llama_guard_ollama
    python -m scripts.score_llama_guard_ollama --limit 200   # smoke test
"""
import argparse
import json
import os
import sys
import time

import requests

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
OUTPUT_FILE = os.path.join("_evidence", "detector_scores", "llama_guard_3_8b.jsonl")
PROGRESS_FILE = os.path.join("_evidence", "llama_guard_8b_scoring_progress.json")

OLLAMA_BASE_URL = "http://localhost:11434/api"
MODEL = "llama-guard3"
REQUEST_TIMEOUT_S = 60
MAX_RETRIES_PER_ROW = 3
RETRY_BACKOFF_S = 5


def load_suite(limit=None):
    rows = []
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def load_already_scored():
    """Returns {id: score} for every row already written -- resumability."""
    scored = {}
    if not os.path.exists(OUTPUT_FILE):
        return scored
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                scored[r["id"]] = r["score"]
            except (json.JSONDecodeError, KeyError):
                # A partial/corrupt last line from a hard kill mid-write.
                # Drop it silently -- the row will simply be rescored.
                continue
    return scored


def ollama_reachable():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def score_one(text):
    """
    Returns 1.0 for unsafe, 0.0 for safe. Raises on any backend failure --
    the caller decides whether to retry.

    Mirrors core/semantic_judge.py::_judge_via_llama_guard's protocol exactly:
    /api/chat (not /api/generate, so Llama Guard's own template applies), a
    single user-role message with no system-prompt contamination, first-line
    parse of the response.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    raw = resp.json().get("message", {}).get("content", "").strip()
    verdict = raw.split("\n")[0].strip().lower()

    if verdict == "safe":
        return 0.0
    if verdict == "unsafe":
        return 1.0
    raise RuntimeError(f"unrecognised verdict: {raw[:120]!r}")


def write_progress(done, total, started_at, last_error=None):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    elapsed = time.time() - started_at
    rate = done / elapsed if elapsed > 0 else 0
    remaining_s = (total - done) / rate if rate > 0 else None
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "done": done, "total": total,
            "elapsed_s": round(elapsed, 1),
            "rate_per_s": round(rate, 4),
            "eta_remaining_s": round(remaining_s, 1) if remaining_s else None,
            "last_error": last_error,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_suite(limit=args.limit)
    already = load_already_scored()
    todo = [r for r in rows if r["id"] not in already]

    print(f"Suite: {len(rows)} rows. Already scored: {len(already)}. Remaining: {len(todo)}.")
    sys.stdout.flush()

    if not todo:
        print("Nothing to do -- all rows already scored.")
        return

    if not ollama_reachable():
        raise SystemExit("Ollama is not reachable at localhost:11434. Start it first.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    started_at = time.time()
    done = len(already)
    total = len(rows)
    consecutive_failures = 0

    # Append mode: never truncates prior progress. Opened once, flushed per
    # row, so a kill -9 loses at most the row currently in flight.
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo):
            last_error = None
            for attempt in range(1, MAX_RETRIES_PER_ROW + 1):
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
                    print(f"  [{i+1}/{len(todo)}] attempt {attempt}/{MAX_RETRIES_PER_ROW} "
                          f"failed: {last_error}")
                    sys.stdout.flush()
                    if attempt < MAX_RETRIES_PER_ROW:
                        time.sleep(RETRY_BACKOFF_S)
            else:
                # All retries exhausted for this row. Record it as unscored
                # (never as a fabricated 0.0) by simply not writing it --
                # the resumability logic will pick it up on a future run.
                print(f"  [{i+1}/{len(todo)}] GIVING UP on row {row['id']} after "
                      f"{MAX_RETRIES_PER_ROW} attempts: {last_error}")
                sys.stdout.flush()

            done += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                write_progress(done, total, started_at, last_error)
                elapsed_m = (time.time() - started_at) / 60
                print(f"  progress: {done}/{total} total ({i+1}/{len(todo)} this run) "
                      f"| {elapsed_m:.1f}m elapsed")
                sys.stdout.flush()

            # If Ollama has gone unreachable for several rows running, stop
            # cleanly rather than burning through retries pointlessly --
            # a relaunch of this same command will resume correctly once
            # Ollama is back.
            if consecutive_failures >= 15:
                print("Ollama appears down (15 consecutive failures). Stopping cleanly; "
                      "rerun this script once Ollama is reachable again -- it will resume.")
                sys.stdout.flush()
                raise SystemExit(1)

    write_progress(done, total, started_at)
    print(f"\nDone. {done}/{total} rows scored -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
