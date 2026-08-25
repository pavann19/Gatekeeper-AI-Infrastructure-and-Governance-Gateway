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


def _tail_raw_lines(
    path: str,
    needed: int | float,
    max_bytes: int | None = None,
    start_pos: int | None = None,
    carry: bytes = b"",
    return_state: bool = False,
):
    """
    Reads up to `needed` non-blank lines from the end of `path` (or starting at
    `start_pos` when resuming a scan), returned newest-first (the order the caller
    wants for an activity feed).

    Returns (lines, truncated, exhausted) or (lines, truncated, exhausted, next_pos, next_carry)
    when return_state=True.
    """
    if max_bytes is None:
        max_bytes = MAX_BYTES_SCANNED  # module-level lookup at call time, not def time

    if not os.path.exists(path):
        if return_state:
            return [], False, True, 0, b""
        return [], False, True

    file_size = os.path.getsize(path)
    if file_size == 0:
        if return_state:
            return [], False, True, 0, b""
        return [], False, True

    lines_newest_first = []
    pos = file_size if start_pos is None else start_pos
    if pos <= 0:
        if return_state:
            return [], False, True, 0, carry
        return [], False, True

    bytes_scanned = 0
    carry_buf = carry  # a partial line straddling this chunk and the previous (newer) one

    with open(path, "rb") as f:
        while pos > 0 and len(lines_newest_first) < needed and bytes_scanned < max_bytes:
            read_size = min(CHUNK_SIZE, pos, max_bytes - bytes_scanned)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            bytes_scanned += read_size

            buffer = chunk + carry_buf
            parts = buffer.split(b"\n")
            if pos > 0:
                # parts[0] is an incomplete line continuing into the
                # not-yet-read, earlier part of the file -- hold it over.
                carry_buf = parts[0]
                complete_parts = parts[1:]
            else:
                carry_buf = b""
                complete_parts = parts

            for raw in reversed(complete_parts):
                if raw.strip():
                    lines_newest_first.append(raw)
                    if len(lines_newest_first) >= needed:
                        break

    exhausted = pos == 0
    truncated = not exhausted and len(lines_newest_first) < needed and bytes_scanned >= max_bytes
    if return_state:
        return lines_newest_first, truncated, exhausted, pos, carry_buf
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
    
    # Pass 1: Cheap attempt reading just enough lines for the requested limit
    raw_lines, truncated, exhausted, pos, carry = _tail_raw_lines(
        audit_path, needed=limit, max_bytes=MAX_BYTES_SCANNED, return_state=True
    )
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

    # Pass 2: Resume-based sweep. If more matching events are needed and the file
    # is not exhausted, continue scanning backward from `pos` (where Pass 1 left off)
    # rather than restarting from EOF.
    if len(events) < limit and not truncated and not exhausted and pos > 0:
        more_lines, truncated2, exhausted2, _, _ = _tail_raw_lines(
            audit_path,
            needed=float("inf"),
            max_bytes=MAX_BYTES_SCANNED,
            start_pos=pos,
            carry=carry,
            return_state=True,
        )
        for raw in more_lines:
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
            if len(events) >= limit:
                break
        scan_truncated = truncated2

    return {"events": events[:limit], "scan_truncated": scan_truncated}


def find_by_request_id(request_id, tenant=None, path=None):
    """
    Returns {"events": [...], "scan_truncated": bool} for every audit
    record carrying this exact `request_id`, in CHRONOLOGICAL order
    (oldest first) -- a trace reads top-to-bottom as a story, the
    opposite order from get_recent_activity's newest-first feed.

    Unlike get_recent_activity, there is no small `limit` to stop early
    at: request_id is a high-cardinality correlation ID (a UUID), so a
    real trace is a handful of lines scattered anywhere in the log, not
    a count that early-exits a tail read the way "give me the last 50"
    does. This always scans the full MAX_BYTES_SCANNED budget looking
    for matches -- `scan_truncated` here means "the trace may be
    incomplete because the byte cap was hit before reaching the start
    of the file," a real possibility for an old request_id on a large
    log, not an edge case to hide.
    """
    audit_path = path or settings.AUDIT_LOG_PATH
    raw_lines, truncated, _exhausted = _tail_raw_lines(audit_path, needed=float("inf"))

    matches = []
    for raw in raw_lines:
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if record.get("request_id") != request_id:
            continue
        record["event_type"] = record.get("event_type") or "legacy"
        record.setdefault("tenant", "unset")
        record.setdefault("capability", "unset")
        if tenant is not None and record["tenant"] != tenant:
            continue
        matches.append(record)

    matches.reverse()  # raw_lines was newest-first; a trace reads oldest-first
    return {"events": matches, "scan_truncated": truncated}
