"""
Dry-runs a CANDIDATE policy file against historical traffic, before it is
deployed (Phase 3, Policy-as-Code roadmap item: "Simulation").

WHAT IT ACTUALLY REPLAYS
--------------------------
Real audit records from AUDIT_LOG_PATH (default: audit.jsonl) -- each
input-assessment record already carries exactly the three inputs
policy_decision() needs: capability, risk, tenant (see
core/logger.py::log_event's schema). This script re-evaluates every one of
those historical (capability, risk, tenant) triples under the CANDIDATE
policy file and diffs the result against what was ACTUALLY decided at the
time, without ever mutating the live policy store core/policy.py's module-
global `_store` represents -- policy_decision()'s `store=` parameter exists
specifically for this.

WHAT IT DOES NOT DO
---------------------
Does not re-run detection. If the candidate policy would only ever be
evaluated against risk levels the fusion pipeline actually produces, this
sees exactly that distribution, for real, from real traffic -- not a
synthetic one. It cannot tell you how a policy change interacts with a
FUTURE detection change; that is a different, undecidable question this
tool does not attempt.

Output-assessment audit records (event_type="output_assessment") carry no
`risk` field and are skipped -- this simulates INPUT policy only, matching
what core/policy.py::policy_decision governs.

Usage:
    python -m scripts.simulate_policy candidate_policy_rules.yaml
    python -m scripts.simulate_policy candidate_policy_rules.yaml --audit-log audit.jsonl
"""
import argparse
import json
import sys
from collections import Counter

from core.config import settings
from core.policy import PolicyStore, policy_decision, validate_policy_file


def load_input_assessment_records(audit_log_path):
    records = []
    try:
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Output-assessment records have no `risk`; skip them rather
                # than guessing -- this tool simulates INPUT policy only.
                if "capability" in row and "risk" in row and "decision" in row:
                    records.append(row)
    except FileNotFoundError:
        print(f"No audit log found at {audit_log_path} -- nothing to simulate against.")
        sys.exit(1)
    return records


def main():
    parser = argparse.ArgumentParser(description="Simulate a candidate policy against historical traffic")
    parser.add_argument("candidate_policy_path")
    parser.add_argument("--audit-log", default=settings.AUDIT_LOG_PATH)
    args = parser.parse_args()

    errors = validate_policy_file(args.candidate_policy_path)
    if errors:
        print(f"REFUSING TO SIMULATE: {args.candidate_policy_path} is invalid:\n")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)

    candidate_store = PolicyStore(path=args.candidate_policy_path)
    candidate_store.load()

    records = load_input_assessment_records(args.audit_log)
    if not records:
        print(f"No input-assessment records found in {args.audit_log}.")
        sys.exit(0)

    print(f"Replaying {len(records)} historical decision(s) from {args.audit_log} "
          f"against {args.candidate_policy_path}...\n")

    changed = []
    unchanged_count = 0
    for row in records:
        capability = row["capability"]
        risk = row["risk"]
        tenant = row.get("tenant", "default")
        old_decision = row["decision"]

        new_decision, reason = policy_decision(capability, risk, tenant, store=candidate_store)

        if new_decision != old_decision:
            changed.append({
                "request_id": row.get("request_id", "unset"),
                "tenant": tenant, "capability": capability, "risk": risk,
                "old_decision": old_decision, "new_decision": new_decision,
                "reason": reason,
            })
        else:
            unchanged_count += 1

    print(f"Unchanged: {unchanged_count} / {len(records)}")
    print(f"Changed:   {len(changed)} / {len(records)}\n")

    if changed:
        transition_counts = Counter(f"{c['old_decision']} -> {c['new_decision']}" for c in changed)
        print("Transitions:")
        for transition, count in transition_counts.most_common():
            print(f"  {transition}: {count}")

        print("\nBy tenant:")
        tenant_counts = Counter(c["tenant"] for c in changed)
        for tenant, count in tenant_counts.most_common():
            print(f"  {tenant}: {count} decision(s) would change")

        print("\nSample of changed decisions (up to 10):")
        for c in changed[:10]:
            print(f"  [{c['tenant']}/{c['capability']}/{c['risk']}] "
                  f"{c['old_decision']} -> {c['new_decision']} "
                  f"(request_id={c['request_id']})")

        stricter = sum(1 for c in changed
                      if {"ALLOW": 0, "RESTRICT": 1, "BLOCK": 2}[c["new_decision"]]
                      > {"ALLOW": 0, "RESTRICT": 1, "BLOCK": 2}[c["old_decision"]])
        looser = len(changed) - stricter
        print(f"\n{stricter} decision(s) become STRICTER, {looser} become LOOSER "
              f"under the candidate policy.")


if __name__ == "__main__":
    main()
