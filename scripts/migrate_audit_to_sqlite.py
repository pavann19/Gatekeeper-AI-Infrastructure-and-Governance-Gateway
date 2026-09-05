"""
One-off migration of the existing JSONL audit trail into the
embedded SQLite store (core/audit_store.py).

    python scripts/migrate_audit_to_sqlite.py
    python scripts/migrate_audit_to_sqlite.py --jsonl audit.jsonl --db audit.db
    python scripts/migrate_audit_to_sqlite.py --dry-run

Properties:
  - Idempotent. A per-line content hash is recorded in a `migrated_line`
    table; re-running skips lines already imported, so it is safe to run
    repeatedly (e.g. after more traffic appends to the JSONL).
  - Pre-`event_type` lines are preserved as `legacy_event` rows with their
    raw JSON intact — never dropped, never mislabelled as one of the four
    current types (matches core/activity.py's "legacy" handling).
  - No content is read or written that the JSONL did not already contain:
    hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import audit_store  # noqa: E402

KNOWN = set(audit_store.KNOWN_EVENT_TYPES)


def _ensure_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS migrated_line ("
        "  line_hash TEXT PRIMARY KEY, imported_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )


def migrate(jsonl_path: str, db_path: str, dry_run: bool = False) -> dict:
    if not os.path.exists(jsonl_path):
        raise SystemExit(f"JSONL audit log not found: {jsonl_path}")

    stats = {"total": 0, "imported": 0, "skipped_dup": 0, "bad_json": 0,
             "legacy": 0, "by_type": {}}

    # autocommit (isolation_level=None): the ledger must not hold an open write
    # transaction across the audit_store.write() call below — that other
    # connection would then block on the write lock ("database is locked").
    ledger = sqlite3.connect(db_path, timeout=5.0, isolation_level=None)
    _ensure_ledger(ledger)
    seen = {r[0] for r in ledger.execute("SELECT line_hash FROM migrated_line")}

    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            h = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if h in seen:
                stats["skipped_dup"] += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue

            et = rec.get("event_type") or "legacy"
            if et not in KNOWN:
                et_for_write = "legacy"
                stats["legacy"] += 1
            else:
                et_for_write = et
            stats["by_type"][et] = stats["by_type"].get(et, 0) + 1

            if not dry_run:
                audit_store.write(et_for_write, rec, path=db_path)
                ledger.execute("INSERT OR IGNORE INTO migrated_line (line_hash) VALUES (?)", (h,))
                seen.add(h)
            stats["imported"] += 1

    ledger.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from core.config import settings
    ap.add_argument("--jsonl", default=settings.AUDIT_LOG_PATH)
    ap.add_argument("--db", default=settings.AUDIT_DB_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and report, write nothing")
    args = ap.parse_args()

    stats = migrate(args.jsonl, args.db, dry_run=args.dry_run)
    print(f"{'DRY RUN — ' if args.dry_run else ''}migrated {args.jsonl} -> {args.db}")
    for k in ("total", "imported", "skipped_dup", "bad_json", "legacy"):
        print(f"  {k:<12} {stats[k]}")
    if stats["by_type"]:
        print("  by event_type:")
        for et, n in sorted(stats["by_type"].items()):
            print(f"    {et:<20} {n}")


if __name__ == "__main__":
    main()
