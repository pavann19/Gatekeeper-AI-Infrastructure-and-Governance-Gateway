"""
Embedded-SQLite mirror of the audit trail.

The audit trail has always been append-only JSONL (`core/logger.py`), read
by a byte-bounded reverse tail scan (`core/activity.py`). That scan is
bounded but a full-budget sweep for a near-zero-match query is still real
work under concurrency, and two deferred-tier items (compliance export,
attack-campaign clustering) are close to unimplementable over a linear
scan. This module adds a queryable, indexed store using **stdlib `sqlite3`
only — no new service, no new deployment dependency**, which is the
property that makes it acceptable given this project's infra reality.

WHAT THIS INCREMENT DOES
------------------------
- `write(event_type, entry)` — mirror one audit record into SQLite. Called
  by `core/logger.py` alongside the existing JSONL write (dual-write).
- `recent(...)` / `by_request_id(...)` — read APIs returning the SAME shape
  as `core/activity.py` so a later increment can swap the reader over with
  no caller change.

FAIL-MODE
-------------------------------------------
The JSONL write stays AUTHORITATIVE for now. `core/logger.py` calls
`write()` inside a `try/except` that logs CRITICAL and swallows — a SQLite
mirror failure does NOT fail the request and does NOT trip
`AUDIT_WRITE_FAILS_CLOSED` (that flag guards the JSONL write, which still
succeeded). Only once reads move to SQLite does a write failure here become
request-fatal.

FOUR EVENT SHAPES STAY SEPARATE
------------------------------
One table per `event_type` — `input_assessment`, `output_assessment`,
`gateway_call`, `tool_call` — for exactly the reason `core/logger.py` keeps
the four functions separate: they answer different questions and carry
non-overlapping fields. A pre-`event_type` JSONL line (migration only) goes
to `legacy_event` with its raw JSON preserved, never dropped.

PRIVACY
-------
Hashes only. The schema has `prompt_hash` / `response_hash` /
`arguments_hash` columns and NO column that could hold raw prompt,
response, or argument text — same guarantee the JSONL trail already makes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Iterable

from core.config import settings
from core.logger import get_logger

logger = get_logger("gatekeeper.audit_store")

# sqlite3 connections are not shareable across threads; open per-call instead.
# This lock only serialises the cheap schema-ensure + insert, so a burst of
# concurrent audit writes cannot interleave a half-created schema.
_write_lock = threading.Lock()

KNOWN_EVENT_TYPES = ("input_assessment", "output_assessment", "gateway_call", "tool_call")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS input_assessment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL DEFAULT 'input_assessment',
    timestamp TEXT, request_id TEXT, tenant TEXT, capability TEXT,
    risk TEXT, decision TEXT, prompt_hash TEXT,
    semantic_score REAL, source TEXT,
    educational_context INTEGER, domain_score REAL,
    symbolic_triggered INTEGER, judge_invoked INTEGER
);
CREATE TABLE IF NOT EXISTS output_assessment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL DEFAULT 'output_assessment',
    timestamp TEXT, request_id TEXT, tenant TEXT, capability TEXT,
    decision TEXT, response_hash TEXT,
    pii_leakage INTEGER, secrets_detected INTEGER, toxicity_detected INTEGER,
    hallucination_detected INTEGER, system_prompt_leak_detected INTEGER,
    source TEXT
);
CREATE TABLE IF NOT EXISTS gateway_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL DEFAULT 'gateway_call',
    timestamp TEXT, request_id TEXT, tenant TEXT, capability TEXT,
    provider TEXT, model TEXT, success INTEGER, latency_ms REAL,
    decision TEXT, usage_json TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS tool_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL DEFAULT 'tool_call',
    timestamp TEXT, request_id TEXT, tenant TEXT, capability TEXT,
    tool TEXT, decision TEXT, risk_level TEXT, reason TEXT,
    arguments_hash TEXT, success INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS legacy_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL DEFAULT 'legacy',
    timestamp TEXT, request_id TEXT, tenant TEXT, raw_json TEXT
);
"""

# request_id / tenant / event_type / timestamp — the four the roadmap names.
_INDEXES = [
    ("input_assessment", "request_id"), ("input_assessment", "tenant"),
    ("input_assessment", "timestamp"), ("input_assessment", "event_type"),
    ("output_assessment", "request_id"), ("output_assessment", "tenant"),
    ("output_assessment", "timestamp"), ("output_assessment", "event_type"),
    ("gateway_call", "request_id"), ("gateway_call", "tenant"),
    ("gateway_call", "timestamp"), ("gateway_call", "event_type"),
    ("tool_call", "request_id"), ("tool_call", "tenant"),
    ("tool_call", "timestamp"), ("tool_call", "event_type"),
    ("legacy_event", "request_id"), ("legacy_event", "timestamp"),
]

# A cross-type view for "recent across everything" without four separate reads.
_VIEW = """
CREATE VIEW IF NOT EXISTS all_events AS
    SELECT id, event_type, timestamp, request_id, tenant, capability, decision
      FROM input_assessment
    UNION ALL
    SELECT id, event_type, timestamp, request_id, tenant, capability, decision
      FROM output_assessment
    UNION ALL
    SELECT id, event_type, timestamp, request_id, tenant, capability, decision
      FROM gateway_call
    UNION ALL
    SELECT id, event_type, timestamp, request_id, tenant, capability, decision
      FROM tool_call
    UNION ALL
    SELECT id, event_type, timestamp, request_id, tenant, NULL, NULL
      FROM legacy_event;
"""

# Column list per table, in insert order (id/event_type are defaulted).
_COLUMNS: dict[str, tuple[str, ...]] = {
    "input_assessment": (
        "timestamp", "request_id", "tenant", "capability", "risk", "decision",
        "prompt_hash", "semantic_score", "source", "educational_context",
        "domain_score", "symbolic_triggered", "judge_invoked"),
    "output_assessment": (
        "timestamp", "request_id", "tenant", "capability", "decision",
        "response_hash", "pii_leakage", "secrets_detected", "toxicity_detected",
        "hallucination_detected", "system_prompt_leak_detected", "source"),
    "gateway_call": (
        "timestamp", "request_id", "tenant", "capability", "provider", "model",
        "success", "latency_ms", "decision", "usage_json", "error"),
    "tool_call": (
        "timestamp", "request_id", "tenant", "capability", "tool", "decision",
        "risk_level", "reason", "arguments_hash", "success", "error"),
}

_BOOL_COLUMNS = {
    "educational_context", "symbolic_triggered", "judge_invoked", "pii_leakage",
    "secrets_detected", "toxicity_detected", "hallucination_detected",
    "system_prompt_leak_detected", "success",
}


def _db_path(path: str | None) -> str:
    return path or settings.AUDIT_DB_PATH


def _connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    for table, col in _INDEXES:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS ix_{table}_{col} ON {table}({col})"
        )
    conn.executescript(_VIEW)
    return conn


def _coerce_row(event_type: str, entry: dict[str, Any]) -> list[Any]:
    """Map an audit `entry` dict to positional values for its table's columns.
    `usage` (gateway) is JSON-encoded into `usage_json`; booleans go to 0/1."""
    values: list[Any] = []
    for col in _COLUMNS[event_type]:
        if col == "usage_json":
            usage = entry.get("usage")
            values.append(json.dumps(usage) if usage is not None else None)
        elif col in _BOOL_COLUMNS:
            v = entry.get(col)
            values.append(None if v is None else int(bool(v)))
        else:
            values.append(entry.get(col))
    return values


def write(event_type: str, entry: dict[str, Any], path: str | None = None) -> None:
    """Mirror one audit record into SQLite. Raises on failure — the caller
    (`core/logger.py`) is responsible for the swallow-and-log fail-mode."""
    with _write_lock:
        conn = _connect(path)
        try:
            if event_type in _COLUMNS:
                cols = _COLUMNS[event_type]
                placeholders = ", ".join(["?"] * len(cols))
                conn.execute(
                    f"INSERT INTO {event_type} ({', '.join(cols)}) VALUES ({placeholders})",
                    _coerce_row(event_type, entry),
                )
            else:
                conn.execute(
                    "INSERT INTO legacy_event (timestamp, request_id, tenant, raw_json) "
                    "VALUES (?, ?, ?, ?)",
                    (entry.get("timestamp"), entry.get("request_id"),
                     entry.get("tenant"), json.dumps(entry, default=str)),
                )
            conn.commit()
        finally:
            conn.close()


# --------------------------------------------------------------------------
# Read APIs — return shape matches core/activity.py exactly:
#   {"events": [...], "scan_truncated": bool}
# SQLite queries are bounded and complete, so scan_truncated is always False.
# --------------------------------------------------------------------------

def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    rec = {k: row[k] for k in row.keys()}
    rec.pop("id", None)
    for col in _BOOL_COLUMNS:
        if col in rec and rec[col] is not None:
            rec[col] = bool(rec[col])
    if "usage_json" in rec:
        raw = rec.pop("usage_json")
        rec["usage"] = json.loads(raw) if raw else None
    if rec.get("event_type") == "legacy" and rec.get("raw_json"):
        base = json.loads(rec["raw_json"])
        base["event_type"] = base.get("event_type") or "legacy"
        base.setdefault("tenant", rec.get("tenant") or "unset")
        base.setdefault("capability", "unset")
        return base
    rec.pop("raw_json", None)
    # Parity with core.activity's JSONL reader.
    rec["event_type"] = rec.get("event_type") or "legacy"
    rec.setdefault("tenant", "unset")
    rec.setdefault("capability", "unset")
    return rec


def recent(
    limit: int = 50,
    tenant: str | None = None,
    event_types: Iterable[str] | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Most-recent audit records, newest first — mirror of
    core.activity.get_recent_activity."""
    limit = max(1, min(int(limit), 200))
    wanted = set(event_types) if event_types else set(KNOWN_EVENT_TYPES) | {"legacy"}
    tables = {
        "input_assessment": "input_assessment", "output_assessment": "output_assessment",
        "gateway_call": "gateway_call", "tool_call": "tool_call", "legacy": "legacy_event",
    }
    conn = _connect(path)
    try:
        rows: list[sqlite3.Row] = []
        for et in wanted:
            table = tables.get(et)
            if table is None:
                continue
            sql = f"SELECT * FROM {table}"
            params: list[Any] = []
            if tenant is not None:
                sql += " WHERE tenant = ?"
                params.append(tenant)
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            rows.extend(conn.execute(sql, params).fetchall())
    finally:
        conn.close()
    records = [_row_to_record(r) for r in rows]
    records.sort(key=lambda r: (r.get("timestamp") or "", r.get("event_type") or ""),
                 reverse=True)
    return {"events": records[:limit], "scan_truncated": False}


def by_request_id(
    request_id: str,
    tenant: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Every audit record for one request_id, oldest first — mirror of
    core.activity.find_by_request_id."""
    conn = _connect(path)
    try:
        rows: list[sqlite3.Row] = []
        for table in ("input_assessment", "output_assessment", "gateway_call",
                      "tool_call", "legacy_event"):
            sql = f"SELECT * FROM {table} WHERE request_id = ?"
            params: list[Any] = [request_id]
            if tenant is not None:
                sql += " AND tenant = ?"
                params.append(tenant)
            rows.extend(conn.execute(sql, params).fetchall())
    finally:
        conn.close()
    records = [_row_to_record(r) for r in rows]
    records.sort(key=lambda r: (r.get("timestamp") or "", r.get("event_type") or ""))
    return {"events": records, "scan_truncated": False}
