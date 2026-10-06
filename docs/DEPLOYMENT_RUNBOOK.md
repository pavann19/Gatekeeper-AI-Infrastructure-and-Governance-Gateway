# Deployment and operations

Docker Compose is the packaged local deployment; native Python supports single-process development. Neither path is evidence of production operation or accepted multi-replica deployment.

## Native Python

Use Python 3.10 with an isolated environment. Install CPU Torch before the API dependencies, then the required privacy model:

```sh
python -m venv .venv-api
# Activate .venv-api before the following commands.
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-api.txt
python -m spacy download en_core_web_sm
python -m scripts.manage_api_keys issue --capability GENERAL --tenant default
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Activate with `source .venv-api/bin/activate` on POSIX or `.venv-api\Scripts\Activate.ps1` on PowerShell. Key issuance prints plaintext once and writes only its hash to the default key store. Set `AUTH_MODE=required` in the environment for attributed access. Native default state files are relative to the working directory. A missing *unconfigured* optional tenant store uses defaults; explicitly configured missing/corrupt tenant stores deny admission.

The broad runtime ranges are not a lockfile. Do not install the canonical replay pins into this environment and assume identical deployment behavior. Downloads require internet access, available model revisions and any gated model authorization. Hardware and model availability determine which detector tier can load.

## Docker Compose

Copy `.env.example` to `.env` and configure a private Grafana password. Provision `config/api_keys.json` and `config/tenants.json` before startup; Compose explicitly configures both paths. For a fresh local anonymous demo, each may be an empty JSON object. Never overwrite existing identity files with empty examples.

The key CLI can provision the mounted store from a Python environment with API dependencies:

```sh
python -m scripts.manage_api_keys --file config/api_keys.json issue --capability GENERAL --tenant default
```

Store plaintext securely and set `AUTH_MODE=required` before public exposure. Set up tenant records for non-default identities, suspension, privacy overrides and limits. Example tenant configuration is shown in [config/README.md](../config/README.md).

```sh
docker compose config --quiet
docker compose up --build
```

Services are Redis, Ollama, the one-shot `model-pull`, `gatekeeper-api`, Prometheus and Grafana. The API waits for Redis and judge-model pull. The operator pages are served inside the API container; there is no separate UI image. Model caches and state persist in named volumes. First startup downloads models and may require substantial memory. Cold or missing models can leave readiness degraded; inspect actual logs and readiness rather than assuming successful model acceptance.

Compose publishes API, Ollama and monitoring ports on the host. Restrict them to the intended network with host/ingress controls. Configure TLS and metrics access for an exposed deployment. Do not expose anonymous assessment, default Grafana credentials, raw provider ports or operator credentials to an untrusted network.

## Health

- `/live`: process responds.
- `/ready`: 503 if policy, symbolic rules, required privacy model, loaded baseline models, configured Redis or persistence probes fail.
- `/health`: dependency diagnostics; HTTP 200 is not an acceptance signal.

Container health checks `/ready`. Readiness does not prove every optional detector tier, judge path or full canonical profile. Confirm the intended deployment profile separately.

## State and policy

| State | Compose storage | Boundary |
|---|---|---|
| Key/tenant stores | `./config` read-only bind mount | Provision/reload separately; credentials stay outside Git/image |
| Audit JSONL/SQLite, reviews, policy snapshots | `audit_data` at `/app/audit` | Backup, access control, capacity and recovery required |
| Exact cache/rate/quota | Redis AOF in `redis_data` | Configured outage denies shared authority |
| Detector model cache | `hf_cache` | Not source or independent model verification |
| Ollama models | `ollama_data` | Access/licensing and model selection required |
| Live action policy | Image-managed `policy_rules.json` | Dynamic writes require a separately provisioned writable path |

`policy_rules.yaml` is a commented equivalent example. `policies/threat_anchors.json` and other `policies/` files supply detector/rule data; they are not the live action policy. Dropping a policy into `config/` has no effect unless `POLICY_RULES_FILE` points there. Configure a writable active-policy directory and coordinated process reload if dynamic deploy/rollback is needed.

`AUDIT_SQLITE_ENABLED` controls SQLite mirroring; `AUDIT_READ_FROM_SQLITE` makes SQLite persistence authoritative for its reads. JSONL write failure is controlled by `AUDIT_WRITE_FAILS_CLOSED`; retain fail-closed operation. These modes are distinct, not an atomic distributed transaction. JSONL multi-writer acceptance is unverified. Review locks require an advisory-lock-capable local filesystem; do not assume network filesystem equivalence.

Local rate/quota budgets are per process. Redis shares those controls and exact-cache state but does not coordinate active policy, review stores, fuzzy indexes or circuit breakers. Run one worker for local state mode. Full application replica acceptance is not established by Redis backend concurrency tests. The supplied Compose API host port also does not define a scalable replica routing setup.

## Operator tools and recovery

Use `scripts.manage_api_keys` for issue/list/revoke, `scripts.validate_policy` for candidate validation, `scripts.simulate_policy` for policy-only replay against historical audit triples, and `scripts.manage_policy_versions` for deploy/list/rollback. Run from the repository root in a configured environment. Simulation does not re-run detection. `scripts.migrate_audit_to_sqlite` retains legacy records during audit migration; inspect its dry-run before importing real data.

Back up identity, audit, review, active policy and snapshots together. Preserve corrupt review files for investigation; restore a verified backup. An expired review cannot be approved. A consumed approval is not automatically retried: a crash can leave external execution uncertain.

For application rollback, restore a known image and compatible configuration, retain named volumes, check readiness and authenticated smoke requests, and inspect errors. Policy rollback is separate. Do not remove volumes during routine recovery. Model/runtime changes require profile-specific validation before reusing measurements.
