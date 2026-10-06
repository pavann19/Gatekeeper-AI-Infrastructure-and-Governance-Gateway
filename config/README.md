# Deployment identity configuration

Compose mounts this directory read-only at `/app/config`. It explicitly configures `api_keys.json` and `tenants.json`; provision both before startup. Missing/unreadable configured tenant state denies admission. These files are local credentials/configuration and must not be committed or copied into the image.

For a fresh local anonymous demo, both files can contain `{}`. Do not overwrite an existing store. Optional authentication then permits GENERAL access where allowed; required authentication denies anonymous callers.

Provision a key with an installed API environment:

```sh
python -m scripts.manage_api_keys --file config/api_keys.json issue --capability GENERAL --tenant default
```

The CLI prints plaintext once and stores its hash. Protect the issued credential. An illustrative tenant record is:

```json
{
  "default": {
    "display_name": "Local application",
    "status": "active"
  }
}
```

Operators can configure tenant rate/quota and privacy settings documented by `core/tenancy.py`. Admission identity comes from keys, not arbitrary request fields. Coordinate reload when changing or suspending identities.

`canonical_replay.json` is the public offline replay profile, not a credential file. The default live policy is image-managed `policy_rules.json` at the repository root. Placing a policy in this directory does not select it: set `POLICY_RULES_FILE` explicitly and provision writable storage if dynamic deploy/rollback is needed. See [deployment](../docs/DEPLOYMENT_RUNBOOK.md) and [security](../docs/SECURITY.md).
