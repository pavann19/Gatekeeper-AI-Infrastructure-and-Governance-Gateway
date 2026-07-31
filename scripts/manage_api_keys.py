"""
API key provisioning for Gatekeeper.

Keys are shown ONCE at creation and stored only as SHA-256 hashes. There is no
command to recover a key: if it is lost, revoke it and issue a new one. That is
deliberate — a key store you can read back is a key store an attacker can read
back.

Usage:
    python -m scripts.manage_api_keys issue --capability ELEVATED --tenant acme
    python -m scripts.manage_api_keys list
    python -m scripts.manage_api_keys revoke --key-id acme-research-01
    python -m scripts.manage_api_keys verify --key gk_xxxxx
"""
import argparse
import json
import os
import sys

from core.auth import VALID_CAPABILITIES, generate_key, hash_key
from core.config import settings


def _load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        raise SystemExit(f"Key store at {path} is unreadable: {e}\n"
                         f"Refusing to overwrite it - fix or move it first.")


def _save(path, store):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    # Best-effort permission tightening. No-op on Windows, which uses ACLs.
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def cmd_issue(args):
    path = args.file
    store = _load(path)

    capability = args.capability.upper()
    if capability not in VALID_CAPABILITIES:
        raise SystemExit(f"capability must be one of {list(VALID_CAPABILITIES)}")

    key_id = args.key_id or f"{args.tenant}-{capability.lower()}-{len(store) + 1:02d}"
    if any(g.get("key_id") == key_id for g in store.values()):
        raise SystemExit(f"key_id {key_id!r} already exists; choose another.")

    plaintext = generate_key()
    store[hash_key(plaintext)] = {
        "capability": capability,
        "tenant": args.tenant,
        "key_id": key_id,
    }
    _save(path, store)

    print("\n" + "=" * 68)
    print("  API KEY ISSUED - copy it now, it will not be shown again")
    print("=" * 68)
    print(f"  key         : {plaintext}")
    print(f"  key_id      : {key_id}")
    print(f"  capability  : {capability}")
    print(f"  tenant      : {args.tenant}")
    print("=" * 68)
    print(f"  Stored (hashed only) in {path}")
    print("  Use as:  Authorization: Bearer <key>")
    if capability == "INTERNAL":
        print("\n  WARNING: INTERNAL maps HIGH -> ALLOW in policy_rules.json.")
        print("  This key bypasses blocking entirely. Issue sparingly.")
    print()


def cmd_list(args):
    store = _load(args.file)
    if not store:
        print(f"No keys in {args.file}. All requests will be anonymous (GENERAL).")
        return
    print(f"{len(store)} key(s) in {args.file}:\n")
    print(f"  {'key_id':<28} {'capability':<12} {'tenant':<16} hash")
    for key_hash, g in sorted(store.items(), key=lambda kv: kv[1].get("key_id", "")):
        print(f"  {g.get('key_id', '?'):<28} {g.get('capability', '?'):<12} "
              f"{g.get('tenant', '?'):<16} ...{key_hash[-12:]}")
    print("\n  Plaintext keys are not recoverable by design.")


def cmd_revoke(args):
    path = args.file
    store = _load(path)
    matches = [h for h, g in store.items() if g.get("key_id") == args.key_id]
    if not matches:
        raise SystemExit(f"No key with key_id {args.key_id!r} in {path}")
    for h in matches:
        del store[h]
    _save(path, store)
    print(f"Revoked {len(matches)} key(s) with key_id {args.key_id!r}.")
    print("Restart the API (or call core.auth.reload_keys()) to apply.")


def cmd_verify(args):
    """Checks a key resolves, without printing or storing it."""
    from core import auth as auth_mod
    from core.auth import KeyStore, resolve_principal

    # resolve_principal reads the module-level store, which is built from
    # settings.API_KEYS_FILE. Point it at --file so this command verifies
    # against the store the operator actually named.
    auth_mod._store = KeyStore(args.file)
    principal = resolve_principal(api_key=args.key)
    print(f"  authenticated : {principal.authenticated}")
    print(f"  capability    : {principal.capability}")
    print(f"  tenant        : {principal.tenant}")
    print(f"  key_id        : {principal.key_id}")
    print(f"  reason        : {principal.reason}")
    sys.exit(0 if principal.authenticated else 1)


def main():
    parser = argparse.ArgumentParser(description="Manage Gatekeeper API keys")
    parser.add_argument("--file", default=settings.API_KEYS_FILE,
                        help=f"Key store path (default {settings.API_KEYS_FILE})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("issue", help="Mint a new API key")
    p.add_argument("--capability", default="GENERAL",
                   help=f"One of {list(VALID_CAPABILITIES)}")
    p.add_argument("--tenant", default="default")
    p.add_argument("--key-id", default=None)
    p.set_defaults(func=cmd_issue)

    p = sub.add_parser("list", help="List key metadata (never the keys)")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("revoke", help="Remove a key by key_id")
    p.add_argument("--key-id", required=True)
    p.set_defaults(func=cmd_revoke)

    p = sub.add_parser("verify", help="Check what a key resolves to")
    p.add_argument("--key", required=True)
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
