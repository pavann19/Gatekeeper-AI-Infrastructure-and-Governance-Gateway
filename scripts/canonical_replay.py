"""Strict frozen-oracle replay with an explicit model/configuration identity.

This is correctness evidence, not an accuracy benchmark or recalibration.
Missing canonical assets are an error; production's optional tiers remain separate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
from unittest.mock import patch

PROFILE = Path("config/canonical_replay.json")


class CanonicalEnvironmentError(RuntimeError):
    pass


def load_profile(path=PROFILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_hash(path):
    # Git may check text out with CRLF on Windows and LF on Linux.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_artifacts(profile):
    for name, expected in profile["artifacts"].items():
        if not Path(name).is_file() or artifact_hash(name) != expected:
            raise CanonicalEnvironmentError(f"Canonical artifact mismatch: {name}")


def validate_configuration(profile, settings):
    for key, expected in profile["settings"].items():
        if getattr(settings, key) != expected:
            raise CanonicalEnvironmentError(f"Canonical configuration mismatch: {key}")


def validate_detectors(profile, registry, verify_weights=True):
    from core.detectors import _verify_detector_integrity
    for name, entry in profile["detectors"].items():
        detector = registry.get(name)
        if detector is None or detector.revision != entry["revision"]:
            raise CanonicalEnvironmentError(f"Canonical detector revision mismatch: {name}")
        available, detail = detector.available()
        if not available:
            raise CanonicalEnvironmentError(f"Canonical required detector unavailable: {name}: {detail}")
        if verify_weights:
            _verify_detector_integrity(name, entry["model_id"], entry["revision"])


def prepare_environment():
    import numpy as np
    import torch
    from core.config import settings
    from core.detectors import get_registry
    from core import embeddings, fusion

    profile = load_profile()
    validate_artifacts(profile)
    validate_configuration(profile, settings)
    for package, expected in profile["runtime_reference"].items():
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected.split("+")[0]:
            raise CanonicalEnvironmentError(f"Canonical runtime mismatch: {package}: {actual} != {expected}")
    random.seed(profile["seed"])
    np.random.seed(profile["seed"])
    torch.manual_seed(profile["seed"])
    torch.set_num_threads(profile["torch_threads"])
    torch.use_deterministic_algorithms(True)
    # A fresh process is required, so an already-loaded floating encoder
    # cannot masquerade as this pinned canonical encoder.
    if embeddings._model is not None:
        raise CanonicalEnvironmentError("Canonical replay requires a fresh model process")
    from sentence_transformers import SentenceTransformer
    embeddings._model = SentenceTransformer(
        profile["embedding"]["model_id"], revision=profile["embedding"]["revision"],
    )
    validate_detectors(profile, get_registry())
    fusion._load_policy()
    if fusion._policy is None or fusion._all_tiers()[0].get("tier_id") != profile["fusion_tier"]:
        raise CanonicalEnvironmentError("Canonical fusion tier mismatch")
    return profile


@contextmanager
def isolated_decisions():
    import core.risk as risk
    from core.circuit_breaker import llama_guard_breaker, ollama_judge_breaker

    def unavailable(*args, **kwargs):
        raise ConnectionError("Canonical judge deliberately unavailable")

    with patch.object(risk, "lookup_cache", return_value=(None, None)), \
         patch.object(risk, "save_cache_entry"), \
         patch.object(risk, "llama_guard_arbitration", return_value=None), \
         patch("core.semantic_judge.requests.post", side_effect=unavailable):
        llama_guard_breaker.reset()
        ollama_judge_breaker.reset()
        yield



def validate_decision_tier(profile, details, case_id):
    if details.get("fusion_available") is False:
        raise CanonicalEnvironmentError(f"Canonical fusion became unavailable: {case_id}")
    if details.get("fusion_available"):
        tier = details.get("fusion_detail") or ""
        if f"tier={profile['fusion_tier']}" not in tier:
            raise CanonicalEnvironmentError(f"Canonical tier changed during replay: {case_id}")

def run_replay(output=None):
    profile = prepare_environment()
    import core.risk as risk
    records = []
    from core.cache import security_cache_scope, _CACHE_SCOPE
    with security_cache_scope("", "default"):
        profile_version = _CACHE_SCOPE.get().split(":")[0]
    with isolated_decisions():
        for row in (json.loads(line) for line in Path("tests/fixtures/decision_replay_corpus.jsonl")
                    .read_text(encoding="utf-8").splitlines() if line.strip()):
            actual, details = risk.assess_risk(row["text"])
            tier = details.get("fusion_detail")
            validate_decision_tier(profile, details, row["id"])
            records.append({
                "case_id": row["id"], "language": row["language"], "attack_class": row["attack_class"],
                "expected_risk": row["expected_risk"], "actual_risk": actual,
                "expected_source": row["expected_source"], "actual_source": details["source"],
                "detector_features": details.get("fusion_detector_scores") or {},
                "cache_hit": False, "policy_capability_context": {"tenant": "default", "capability": None},
                "security_configuration_version": profile_version,
                "judge_state": profile["judge_state"], "model_tier": tier or "symbolic/fast-path",
                "details": details,
            })
    mismatches = [r for r in records if r["actual_risk"] != r["expected_risk"]]
    print(f"Canonical rows scored: {len(records)}", flush=True)
    summary = {"cases": len(records), "mismatches": len(mismatches),
               "high_to_low": sum(r["expected_risk"] == "HIGH" and r["actual_risk"] == "LOW"
                                  for r in mismatches), "profile": profile,
               "runtime": {p: importlib.metadata.version(p) for p in profile["runtime_reference"]}}
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "decisions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8",
        )
        (output / "mismatches.json").write_text(json.dumps(mismatches, indent=2, ensure_ascii=False), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        fields = [k for k in records[0] if k != "details"]
        with (output / "mismatch-table.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in mismatches:
                writer.writerow({k: json.dumps(record[k], sort_keys=True) if isinstance(record[k], dict)
                                 else record[k] for k in fields})
    return summary, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Private evidence directory outside the public checkout")
    args = parser.parse_args()
    summary, _ = run_replay(args.output)
    print(json.dumps({k: summary[k] for k in ("cases", "mismatches", "high_to_low")}))
    return int(bool(summary["mismatches"]))


if __name__ == "__main__":
    raise SystemExit(main())
