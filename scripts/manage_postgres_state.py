"""Explicit schema preflight/migration; never run automatically on requests."""
import argparse

from core.postgres_state import get_postgres_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'migrate'))
    parser.add_argument('--confirm-empty-or-known-database', action='store_true')
    args = parser.parse_args()
    store = get_postgres_store()
    if args.action == 'migrate':
        if not args.confirm_empty_or_known_database:
            parser.error('Migration requires explicit --confirm-empty-or-known-database')
        store.migrate()
    ready = store.ready()
    print('PASS: state schema ready' if ready else 'FAIL: state schema unavailable')
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
