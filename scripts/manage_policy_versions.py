"""
Operator CLI for policy versioning and rollback (Phase 3, Policy-as-Code).
See core/policy_versioning.py for the design rationale.

Usage:
    python -m scripts.manage_policy_versions list
    python -m scripts.manage_policy_versions deploy candidate_policy_rules.yaml
    python -m scripts.manage_policy_versions rollback 20260821T140000Z__ab12cd34ef56.json
"""
import sys

from core.policy import validate_policy_file
from core.policy_versioning import deploy_policy, list_versions, rollback_to


def cmd_list():
    versions = list_versions()
    if not versions:
        print("No policy versions saved yet.")
        return
    print(f"{len(versions)} version(s), newest first:\n")
    for v in versions:
        print(f"  {v}")


def cmd_deploy(new_path):
    errors = validate_policy_file(new_path)
    if errors:
        print(f"REFUSING TO DEPLOY: {new_path} is invalid:\n")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)

    previous = deploy_policy(new_path)
    print(f"Deployed {new_path} as the live policy.")
    if previous:
        print(f"Previous policy saved as version: {previous}")
        print(f"Roll back with: python -m scripts.manage_policy_versions rollback {previous}")
    else:
        print("(No previous policy existed to snapshot — this was the first deploy.)")


def cmd_rollback(version_name):
    try:
        rollback_to(version_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'python -m scripts.manage_policy_versions list' to see available versions.")
        sys.exit(1)
    print(f"Rolled back to version: {version_name}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    command = sys.argv[1]
    if command == "list":
        cmd_list()
    elif command == "deploy" and len(sys.argv) == 3:
        cmd_deploy(sys.argv[2])
    elif command == "rollback" and len(sys.argv) == 3:
        cmd_rollback(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
