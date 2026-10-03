# Policy Files

Gatekeeper has one live decision policy file and several supporting policy-data
files. This split is intentional so the public repository does not tell two
different stories about which file controls runtime decisions.

## Live Decision Policy

`policy_rules.json` is the default runtime policy loaded by
`POLICY_RULES_FILE`. It maps tenant, capability tier, and risk level to an
action such as `ALLOW`, `RESTRICT`, or `BLOCK`.

`policy_rules.yaml` uses the same schema and exists as a commented
Policy-as-Code example. Operators can point `POLICY_RULES_FILE` at a YAML file,
but the repository default remains `policy_rules.json`.

## Threat Taxonomy Data

`policies/threat_anchors.json` is not the live decision policy. It contains
semantic threat-anchor text grouped by attack class for detector scoring and
threat taxonomy coverage.

The other files under `policies/` are supporting detector and classifier data:

| file | purpose |
|---|---|
| `policies/symbolic_rules.json` | Regex and keyword rules for symbolic vetoes |
| `policies/domain_anchors.json` | Domain/topicality anchor corpus |
| `policies/domain_corpus.json` | Domain classifier corpus |
| `policies/meta_intent_anchors.json` | Meta-intent anchor examples |

## Deployment Note

`config/` is for deployment-time optional files such as API keys and tenants.
Dropping a policy file into `config/` does not change the live policy unless
`POLICY_RULES_FILE` is explicitly pointed there.
