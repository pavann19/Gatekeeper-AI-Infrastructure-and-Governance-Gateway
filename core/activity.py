"""
Read access over the audit log (`AUDIT_LOG_PATH`, one JSON object per line,
written by core/logger.py's four log_*_event functions) for the client
UI's activity feed (Phase 7).

This module only READS. Nothing here writes to the audit log or changes
what gets recorded -- it exists purely to answer "what happened recently",
the same question an operator would otherwise answer by `tail -f
audit.jsonl` by hand.

WHY A REVERSE-CHUNK TAIL READ, NOT `open(path).readlines()`
--------------------------------------------------------------
The audit log is append-only and, in a real deployment, grows without
bound. "Show me the last 50 events" should not require reading a
multi-gigabyte file into memory -- this reads from the END of the file
backward, in fixed-size chunks, stopping as soon as enough MATCHING lines
are found. `MAX_BYTES_SCANNED` bounds the worst case (a filter that
matches almost nothing) so a query can never turn into an unbounded full
file scan; hitting that cap is reported back (`scan_truncated`) rather
than silently returning fewer results than asked for and looking like
"that's all there is."

WHY event_type CAN BE MISSING ON OLDER LINES
--------------------------------------------------------------
`event_type` and today's `tenant`/`capability` fields were added over the
course of this project (see log_event's own docstring on the
"unset" vs "default" tenant distinction). A line written before that
schema existed is still a real historical record and is not discarded --
it is surfaced with event_type normalised to "legacy" rather than being
silently dropped or mislabelled as one of today's four types.
"""
from __future__ import annotations

import json
import os

from core.config import settings

CHUNK_SIZE = 65536
MAX_BYTES_SCANNED = 20_000_000  # 20MB -- bounds a filter that matches almost nothing
DEFAULT_LIMIT = 50
MAX_LIMIT = 200

KNOWN_EVENT_TYPES = ("input_assessment", "output_assessment", "gateway_call", "tool_call")


def _tail_raw_lines(path: str, needed: int, max_bytes: int = None):
    """
    Reads up to `needed` non-blank lines from the end of `path`, returned
    newest-first (the order the caller wants for an activity feed).

    Returns (lines, truncated, exhausted):
      - truncated: `max_bytes` was hit before `needed` lines were found AND
        before the start of the file was reached -- there may be more
        matching lines this call simply didn't scan far enough to see.
      - exhausted: the scan reached byte 0 of the file -- every line that
        exists has now been read, however few that turned out to be. A
        caller retrying with a larger `needed` after `exhausted=True` would
        get back the exact same lines, not more.
    """
    if max_bytes is None:
        max_bytes = MAX_BYTES_SCANNED  # module-level lookup at call time, not def time

    if not os.path.exists(path):
        return [], False, True

    file_size = os.path.getsize(path)
    if file_size == 0:
        return [], False, True

    lines_newest_first = []
    pos = file_size
    bytes_scanned = 0
    carry = b""  # a partial line straddling this chunk and the previous (newer) one

    with open(path, "rb") as f:
        while pos > 0 and len(lines_newest_first) < needed and bytes_scanned < max_bytes:
            read_size = min(CHUNK_SIZE, pos, max_bytes - bytes_scanned)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            bytes_scanned += read_size

            buffer = chunk + carry
            parts = buffer.split(b"\n")
            if pos > 0:
                # parts[0] is an incomplete line continuing into the
                # not-yet-read, earlier part of the file -- hold it over.
                carry = parts[0]
                complete_parts = parts[1:]
            else:
                carry = b""
                complete_parts = parts

            for raw in reversed(complete_parts):
                if raw.strip():
                    lines_newest_first.append(raw)
                    if len(lines_newest_first) >= needed:
                        break

    exhausted = pos == 0
    truncated = not exhausted and len(lines_newest_first) < needed
    return lines_newest_first, truncated, exhausted


def get_recent_activity(limit=DEFAULT_LIMIT, tenant=None, event_types=None, path=None):
    """
    Returns {"events": [...], "scan_truncated": bool}, newest first.

    `tenant`: exact-match filter, or None for no tenant filter (the caller
    -- api/main.py -- is responsible for deciding whether an untenanted
    query is allowed; this function has no notion of capability).
    `event_types`: an iterable of event_type strings to include, or None
    for all types (known and "legacy").
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    audit_path = path or settings.AUDIT_LOG_PATH
    event_type_filter = set(event_types) if event_types else None

    events = []
    scan_truncated = False
    needed_raw = limit
    # A filter can reject most lines read, so we may need several passes
    # over successively larger tail windows before we have `limit` MATCHING
    # events or hit MAX_BYTES_SCANNED -- each pass re-reads from the end
    # rather than resuming mid-scan, which is simpler and still bounded by
    # the same MAX_BYTES_SCANNED cap overall.
    while True:
        raw_lines, truncated, exhausted = _tail_raw_lines(audit_path, needed_raw)
        events = []
        for raw in raw_lines:
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            record["event_type"] = record.get("event_type") or "legacy"
            record.setdefault("tenant", "unset")
            record.setdefault("capability", "unset")

            if tenant is not None and record["tenant"] != tenant:
                continue
            if event_type_filter is not None and record["event_type"] not in event_type_filter:
                continue
            events.append(record)

        scan_truncated = truncated
        # Stop once there are enough matches, the byte cap was hit (nothing
        # more to safely read), or the whole file has been read (retrying
        # with a bigger needed_raw cannot produce more lines than exist).
        if len(events) >= limit or truncated or exhausted:
            break
        needed_raw *= 4

    return {"events": events[:limit], "scan_truncated": scan_truncated}
