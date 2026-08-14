# Config directory

Mounted read-only into the `gatekeeper-api` container at `/app/config`
(see `docker-compose.yml`). Everything in here is optional — an operator
who deploys without touching this directory gets exactly the same
zero-trust defaults as running Gatekeeper outside Docker:

| File | Absent means | See |
|---|---|---|
| `api_keys.json` | Every caller is anonymous, served at GENERAL (least privilege) | `core/auth.py` |
| `tenants.json` | Every caller resolves to the active `DEFAULT_TENANT` | `core/tenancy.py` |
Deliberately **not listed above**: `policy_rules.json`. Unlike the two
files in the table, an absent policy file is not a safe default —
`core/policy.py` fails closed to BLOCK for every single request without a
usable `tenants.default` entry. `docker-compose.yml` therefore does NOT
point `POLICY_RULES_FILE` at this directory; it keeps `core/config.py`'s
default, which resolves to the copy the Dockerfile bakes in from the repo
root (`COPY . .`) — already the correct, tenant-scoped policy this project
ships with. To override the shipped policy, edit `POLICY_RULES_FILE` in
`docker-compose.yml` yourself and place your replacement here; do not do
it by just dropping a file in this directory, since nothing reads it from
here by default.

`api_keys.json` and `tenants.json` are not committed to this repo (see
`.gitignore` / `.dockerignore`) — provisioning them is a deployment-time
decision, not a build-time one.
