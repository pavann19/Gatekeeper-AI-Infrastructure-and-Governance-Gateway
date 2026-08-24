"""
Read access over this project's own benchmark result files (Phase 7,
Developer UI: "Benchmarks") -- surfaces the SAME accuracy/latency/
confusion-matrix JSON reports this project's own benchmark scripts
produce and commit as evidence (`_evidence/benchmark_results_*.json`),
not a re-run or a re-derived summary. If a number shown here looks wrong,
the fix is in whatever benchmark script produced the underlying file, not
in this module -- this module only reads and lists.

WHY ONLY THE `benchmark_results_*.json` SHAPE
--------------------------------------------------------------
`_evidence/` holds several genuinely different report types (calibration
curves, detector comparisons, ensemble analyses, ...) accumulated across
this project's history. Only `benchmark_results_*.json` files share one
consistent shape (`config`/`dataset`/`cold`/`warm` with matching metrics
underneath both) that a single UI can render meaningfully without
per-report-type special-casing. Other report types are real evidence too,
just not surfaced by THIS view yet -- narrower scope now, not a claim
that the others don't matter.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

from core.config import settings

FILENAME_PATTERN = "benchmark_results_*.json"


def list_benchmark_runs(directory=None) -> dict:
    """
    Returns {"runs": [...], "errors": [...]}. Each run is the file's own
    parsed JSON content plus `_filename` and `_mtime` (ISO 8601, for
    sorting/display) -- newest first. A file that fails to parse is
    reported by name in `errors` rather than silently excluded, so a
    corrupt evidence file is visible as a problem, not invisible as if it
    never existed.

    Returns an empty result (not an error) when the directory itself
    doesn't exist -- a fresh checkout or a deployment that never ran a
    benchmark has nothing to show, which is a normal state, not a fault.
    """
    directory = directory or settings.EVIDENCE_DIR
    runs = []
    errors = []

    if not os.path.isdir(directory):
        return {"runs": [], "errors": []}

    for path in sorted(glob.glob(os.path.join(directory, FILENAME_PATTERN))):
        filename = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            errors.append({"filename": filename, "error": str(e)})
            continue

        if not isinstance(content, dict):
            errors.append({"filename": filename, "error": "expected a JSON object at the top level"})
            continue

        mtime = os.path.getmtime(path)
        run = dict(content)
        run["_filename"] = filename
        run["_mtime"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        runs.append((mtime, run))

    runs.sort(key=lambda pair: pair[0], reverse=True)
    runs = [run for _mtime, run in runs]
    return {"runs": runs, "errors": errors}
