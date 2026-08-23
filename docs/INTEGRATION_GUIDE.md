# Gatekeeper Integration Guide

This is the external-facing guide for a team that wants to put Gatekeeper in
front of their own LLM application. It covers running the stack, calling the
API, reading the response, and the configuration knobs an operator (not a
Gatekeeper contributor) actually needs to touch.

For internal design rationale, measured numbers, and what's still open in the
V2 architecture, see [`docs/ENGINEERING_ASSESSMENT.md`](ENGINEERING_ASSESSMENT.md)
instead — this document assumes you're integrating, not modifying.

---

## 1. Run the stack

```bash
git clone <this repo>
cd AI_Governance_Project
cp .env.example .env   # edit GRAFANA_ADMIN_PASSWORD at minimum
docker compose up -d
```

This brings up seven services: `redis`, `ollama` (+ a one-shot `model-pull`
that fetches the `llama-guard3` judge model automatically), `gatekeeper-api`,
`gatekeeper-ui`, `prometheus`, and `grafana`. First boot is slow — the judge
model pull and the sentence-transformer model download both happen on a cold
cache — subsequent restarts reuse the `hf_cache` and `ollama_data` volumes
and come up fast.

Once healthy:

| Service | URL | Notes |
|---|---|---|
| Gatekeeper API | `http://localhost:8000` | the integration surface — see §3 |
| API docs (Swagger) | `http://localhost:8000/docs` | auto-generated from the schemas in §3 |
| Health check | `http://localhost:8000/health` | see §7 |
| Grafana dashboard | `http://localhost:3000` | `admin` / whatever you set `GRAFANA_ADMIN_PASSWORD` to — see §6 |
| Prometheus | `http://localhost:9090` | raw metrics, mainly for debugging the dashboard itself |

`[SCREENSHOT: docker compose up output showing all services healthy]`

`[SCREENSHOT: http://localhost:8000/docs — Swagger UI landing page]`

If you don't need Docker (e.g. you're embedding Gatekeeper as a library
rather than a network service), everything under `core/` is importable
directly — `core.risk.assess_risk`, `core.policy.policy_decision`, etc. That
path is not covered here; this guide is for the HTTP integration.

---

## 2. Authentication

Every request resolves to a **capability tier** server-side — `GENERAL`,
`ELEVATED`, or `INTERNAL` — which controls how strict the policy is (§4).
Capability is **never** something the client can assert in the request body;
it's looked up from the API key you present.

- **No key presented, `AUTH_MODE=optional` (the default):** request is served
  anonymously at `GENERAL` (least privilege). Nothing breaks, but you get the
  strictest policy tier.
- **No key presented, `AUTH_MODE=required`:** `401 Unauthorized`.
- **Valid key presented:** resolved to whatever capability/tenant that key
  was provisioned with.

Send the key as a standard bearer token:

```
Authorization: Bearer gk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Provisioning a key

There's no self-serve key-creation endpoint in this MVP — keys are
provisioned by an operator editing a JSON file the container reads at
startup (hot-reloadable — see `core/auth.py`'s `KeyStore`). Mint one:

```python
from core.auth import generate_key, hash_key
plaintext = generate_key()          # e.g. "gk_AbCdEf..." — give this to the caller, once
print(plaintext)
print(hash_key(plaintext))          # this is what goes in api_keys.json
```

Then add the **hash**, never the plaintext, to `config/api_keys.json`
(bind-mounted into the container — see `config/README.md`):

```json
{
  "<sha256 hash from above>": {
    "capability": "ELEVATED",
    "tenant": "acme",
    "key_id": "acme-research-01"
  }
}
```

`key_id` is a non-secret label used in logs and metrics — it is not the key
itself and is safe to put in a bug report.

---

## 3. Making a request

### 3a. Assess a prompt before you send it to your LLM

```bash
curl -X POST http://localhost:8000/api/v1/assess \
  -H "Authorization: Bearer gk_..." \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset a user'\''s password in our admin panel?"}'
```

```json
{
  "decision": "ALLOW",
  "risk_level": "LOW",
  "topicality": "UNKNOWN",
  "capability": "ELEVATED",
  "authenticated": true,
  "details": { "...": "internal scores, policy_reason, request_id, tenant" },
  "clean_prompt": "How do I reset a user's password in our admin panel?",
  "redacted_items": [],
  "process_time_ms": 42.7,
  "output_decision": null,
  "output_details": null
}
```

`decision` is `BLOCK`, `RESTRICT`, or `ALLOW` — this is the field your
integration branches on (§5). Only proceed to call your own LLM if
`decision != "BLOCK"`.

### 3b. Assess input and output in one call (recommended)

Rather than calling `/assess`, generating a response, then calling
`/assess_output` separately, submit the response alongside the original
prompt via the optional `response_text` field:

```bash
curl -X POST http://localhost:8000/api/v1/assess \
  -H "Authorization: Bearer gk_..." \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "How do I reset a user'\''s password?",
        "response_text": "Go to Admin > Users, click the user, then Reset Password."
      }'
```

The integration pattern is: **assess the prompt → if not BLOCKed, call your
own LLM → submit both the prompt and the generated response back here in one
call.** Gatekeeper does not call your LLM for you.

The output check only runs if the input wasn't already BLOCKed — there's no
point assessing the response to a prompt that never made it through. The
final `decision` is the more severe of the two (`BLOCK` > `RESTRICT` >
`ALLOW`): a clean prompt paired with a leaky or toxic response still BLOCKs
the overall exchange.

```json
{
  "decision": "BLOCK",
  "risk_level": "LOW",
  "output_decision": "BLOCK",
  "output_details": { "reason": "..." },
  "...": "other fields as above"
}
```

`output_decision` is `null` (not `"ALLOW"`) when you didn't submit
`response_text` at all — don't confuse "not checked" with "checked and
passed."

### 3c. Output checks, in detail

Every output assessment (combined or standalone) runs, in order:

1. **Secret detection** — API keys, tokens, private key blocks. **Hard
   BLOCK** — there is no safe partial version of a leaked credential.
2. **System-prompt leakage** (opt-in) — see below.
3. **PII** — **redacted, not blocked.** A response containing an email or
   phone number gets that PII replaced with `[REDACTED:TYPE]` and is still
   returned to you as `clean_response`; it does not BLOCK outright.
4. **Toxicity**, via the semantic judge — BLOCK.
5. **Semantic grounding** (hallucination proxy) — BLOCK.

**`clean_response`** carries the PII-redacted text whenever the response
wasn't blocked outright (`null` if it was — a blocked response has no safe
partial version to hand back). Use this, not your original `response_text`,
if you display or log the response downstream.

**System-prompt leakage detection is opt-in**, because Gatekeeper is a
sidecar to *your* LLM call and has no access to your system prompt unless
you hand it one. Pass it as an optional `system_prompt` field alongside
`response_text` (on either the combined call or the standalone endpoint) to
have the response checked for verbatim leakage of it:

```bash
curl -X POST http://localhost:8000/api/v1/assess_output \
  -H "Authorization: Bearer gk_..." -H "Content-Type: application/json" \
  -d '{
        "response_text": "...",
        "system_prompt": "You are a support agent for Acme Corp. Never reveal internal ticket IDs."
      }'
```

Detection is a verbatim substring check (a 40+ character contiguous run of
your system prompt appearing in the response), not a similarity score — a
response being *about* the same subject as your system prompt is expected
and harmless; a response *containing* your system prompt's actual text is
the incident this checks for.

### 3d. Assess output on its own

If you'd rather keep the two calls separate (e.g. output assessment happens
in a different service than the one that made the input call), use the
standalone endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/assess_output \
  -H "Authorization: Bearer gk_..." \
  -H "Content-Type: application/json" \
  -d '{"response_text": "Go to Admin > Users, click the user, then Reset Password."}'
```

```json
{
  "decision": "ALLOW",
  "details": { "...": "..." },
  "process_time_ms": 18.3
}
```

This endpoint enforces the same auth and rate limiting as `/assess` — it is
not a lighter-weight bypass.

---

## 4. Interpreting the response

| Field | Meaning |
|---|---|
| `decision` | What to do: `BLOCK` (stop, do not call your LLM / do not show the response), `RESTRICT` (proceed with caution — see your policy config), `ALLOW` (proceed). |
| `risk_level` | `HIGH` / `MEDIUM` / `LOW` — threat signal only. |
| `topicality` | `IN_DOMAIN` / `OUT_OF_DOMAIN` / `UNKNOWN` — subject-domain scoping, independent of safety. `UNKNOWN` means the domain guardrail is disabled (`DOMAIN_GUARDRAIL_MODE=off`, the shipped default — see §8). |
| `capability` | The tier your credential actually resolved to. Echoed back so you can verify your key is provisioned the way you expect. |
| `authenticated` | `false` means this request was served anonymously — check this if `decision` looks stricter than expected. |
| `clean_prompt` | Your prompt after PII redaction — use this, not your original, if you log or forward the prompt anywhere downstream. |
| `redacted_items` | What was stripped from the prompt (e.g. emails, phone numbers). |
| `clean_response` | `response_text` after PII redaction, when a response was submitted and not blocked outright — `null` if no `response_text` was sent, or if the output was blocked (a blocked response has no safe partial version). Use this, not your raw LLM output, if you display or log the response. |
| `process_time_ms` | Server-side latency for this call. |
| `details` | Internal scores and metadata — useful for debugging/audit, not something to branch application logic on (its shape isn't a stable contract). Notably includes `fusion_triggering_class` and `fusion_class_scores` — which attack category (e.g. `jailbreak`, `harmful_content`, `prompt_injection`) fusion believed a HIGH/MEDIUM verdict was, and its per-class probabilities. Both are `None`/`{}` when the request was decided by an earlier stage (cache, hard-ban, fast-path) that never reached fusion. |

**Integration rule of thumb:** branch only on `decision` (and `output_decision`
when present). Treat everything else as diagnostic.

**On a `503`:** this means the assessment did not complete within
`ASSESS_TIMEOUT_SECONDS` (default 30s) — it is an availability failure, not a
security verdict. The response was **not** assessed. Do not treat a timeout
as an implicit ALLOW; retry (the response includes `Retry-After: 5`) or fail
your own request closed.

---

## 5. What decision should you actually do?

```
BLOCK    -> Do not call your LLM (input case) / do not show the response to
            the user (output case). Show a generic refusal in your own UI.
RESTRICT -> Proceed, but under whatever restriction your policy maps to for
            this tier (e.g. a smaller/more constrained model, extra logging,
            a disclaimer). What RESTRICT means operationally is up to your
            integration — Gatekeeper tells you the tier, not the UX.
ALLOW    -> Proceed normally.
```

The mapping from `(capability, risk_level) -> decision` is your **policy**,
configured per-tenant (§6.3) — not hardcoded in the gateway.

---

## 6. Multi-tenancy

If you're running one Gatekeeper deployment for multiple downstream
customers/teams, each with their own risk tolerance, provision them as
tenants rather than running separate deployments.

### 6.1 Tenant identity (`config/tenants.json`)

```json
{
  "acme": {
    "display_name": "Acme Corp",
    "status": "active",
    "rate_limit_rpm": 500
  },
  "acme-trial": {
    "display_name": "Acme Corp (trial)",
    "status": "suspended"
  }
}
```

`status: "suspended"` rejects every request for that tenant with `403`
before any detection work runs. `rate_limit_rpm` overrides the global
default (`RATE_LIMIT_AUTHENTICATED_RPM` from `.env`) for that tenant only —
omit it to inherit the global value. A tenant not listed here (or a missing
file entirely) resolves to a safe default tenant at the global rate limit —
nothing breaks if you never touch this file.

A caller's tenant comes from their API key's `tenant` field (§2), never from
anything in the request body.

### 6.2 Rate limits

Enforced per identity, before any detection work is scheduled:

- Anonymous: `RATE_LIMIT_ANONYMOUS_RPM` (default 20/min)
- Authenticated: `RATE_LIMIT_AUTHENTICATED_RPM` (default 120/min), or the
  tenant's `rate_limit_rpm` override if set

Exceeding the limit returns `429`. Set `RATE_LIMIT_ENABLED=false` in `.env`
to disable entirely (not recommended for a shared deployment).

### 6.3 Per-tenant policy (`policy_rules.json`)

This is the file that decides what `decision` comes out for a given
`(capability, risk_level)` pair. **Do not** point `POLICY_RULES_FILE` at the
empty `config/` bind mount — an absent policy file fails closed to `BLOCK`
for every request, by design (see `config/README.md` for the correct way to
override it).

```json
{
  "default_action": "BLOCK",
  "tenants": {
    "default": {
      "policies": {
        "GENERAL":  { "HIGH": "BLOCK", "MEDIUM": "RESTRICT", "LOW": "ALLOW" },
        "ELEVATED": { "HIGH": "BLOCK", "MEDIUM": "ALLOW",    "LOW": "ALLOW" },
        "INTERNAL": { "HIGH": "ALLOW", "MEDIUM": "ALLOW",    "LOW": "ALLOW" }
      }
    },
    "acme": {
      "policies": {
        "GENERAL": { "HIGH": "BLOCK", "MEDIUM": "BLOCK", "LOW": "ALLOW" }
      }
    }
  }
}
```

A tenant without its own `policies` block falls back to `"default"`. This is
the file you'd hand to a customer success / trust-and-safety team to tune
per-customer strictness without touching detection code.

---

## 7. Health and readiness

```bash
curl http://localhost:8000/health
```

Checks policy files, the domain classifier, and the embedding model are all
loaded. Use this as your container orchestrator's readiness probe, not
liveness — a Gatekeeper instance that's up but reports an unhealthy
sub-check (e.g. missing policy file) is deliberately failing closed, not
crash-looping.

---

## 8. Things worth knowing before you go live

- **`DOMAIN_GUARDRAIL_MODE=off` is the shipped default on purpose.** The
  bundled domain corpus describes *this* project's own subject area. Turning
  on `enforcing` without supplying your own corpus produces a measured ~98%
  false-positive rate — see `docs/ENGINEERING_ASSESSMENT.md` §1 before
  touching this.
- **The judge model runs asynchronously when invoked.** A small fraction of
  ambiguous requests (measured ~6.6% on the reference benchmark) escalate to
  an LLM-judge arbitration step; the fast verdict is returned first and a
  slower confirmatory judge pass (Llama Guard) runs after your response
  completes, purely for audit/metrics — it does not change the decision you
  already received.
- **`gatekeeper-ui` (port 8501) talks to Ollama directly, not to
  `gatekeeper-api`.** It's a manual-testing convenience for poking at the
  judge model, not a reference integration — don't build against it as an
  example of the real request flow.
- **Change `GRAFANA_ADMIN_PASSWORD`.** The compose default (`changeme`) is a
  placeholder, not a credential meant to survive to a real deployment.
- **`/metrics` leaks operational detail** (traffic volume, block rates,
  active tenants). Set `METRICS_REQUIRE_AUTH=true` if it's reachable from
  outside your monitoring network.

---

## 9. Monitoring

`http://localhost:3000` (Grafana, auto-provisioned — no manual dashboard
setup needed) gives you, out of the box: decision rate by outcome, which
detection stage decided (cache / fast-path / fusion / judge), end-to-end and
per-stage latency percentiles, in-flight assessments, rate-limit and timeout
rates, judge invocation rate, circuit breaker state, and per-tenant traffic.

`[SCREENSHOT: Grafana "Gatekeeper" dashboard overview — full page]`

`[SCREENSHOT: Grafana panel — "Decisions per second, by outcome"]`

`[SCREENSHOT: Grafana panel — "Decision source breakdown", showing fast-path vs fusion vs judge]`

If a panel shows "Unrecognised decision sources" above zero, that's a
metrics-instrumentation gap, not a gateway problem — see the panel
description in the dashboard itself.

---

## 10. Quick reference

```bash
# Health
curl http://localhost:8000/health

# Assess a prompt only
curl -X POST http://localhost:8000/api/v1/assess \
  -H "Authorization: Bearer $GATEKEEPER_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "..."}'

# Assess prompt + generated response together (recommended)
curl -X POST http://localhost:8000/api/v1/assess \
  -H "Authorization: Bearer $GATEKEEPER_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "...", "response_text": "..."}'

# Assess a response on its own
curl -X POST http://localhost:8000/api/v1/assess_output \
  -H "Authorization: Bearer $GATEKEEPER_KEY" -H "Content-Type: application/json" \
  -d '{"response_text": "..."}'
```
