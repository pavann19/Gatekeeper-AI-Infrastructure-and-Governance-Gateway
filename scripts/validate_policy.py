"""
Validation step for a candidate policy file, before it is deployed (Phase 3,
Policy-as-Code). Supports both the existing JSON format and the new YAML
one (core/policy.py::_parse_policy_file dispatches on extension).

WHY THIS IS A SEPARATE TOOL FROM PolicyStore.load()
-----------------------------------------------------
The live loader's job is availability: one malformed tenant entry must not
take down every OTHER tenant's policy, so it logs a warning and drops just
that entry. That is the wrong behaviour for an operator checking a file
BEFORE it goes anywhere near the running gateway -- they want every problem
at once, not one warning discovered by trial and error after each fix. This
script uses core.policy.validate_policy_file, which does exactly that.

Usage:
    python -m scripts.validate_policy policy_rules.yaml
    python -m scripts.validate_policy policy_rules.json
"""
import sys

from core.policy import validate_policy_file


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.validate_policy <path-to-policy-file>")
        sys.exit(2)

    path = sys.argv[1]
    errors = validate_policy_file(path)

    if not errors:
        print(f"OK: {path} is a valid policy file.")
        sys.exit(0)

    print(f"INVALID: {path} has {len(errors)} problem(s):\n")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    sys.exit(1)


if __name__ == "__main__":
    main()
