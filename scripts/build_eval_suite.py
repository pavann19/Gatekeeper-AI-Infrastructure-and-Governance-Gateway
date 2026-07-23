"""
Builds the taxonomy-labeled, multi-source evaluation suite.

WHY THIS EXISTS
---------------
The project previously published metrics from a single 546-prompt dataset. That
is inadequate for three reasons, all of which this script addresses:

1. TAXONOMY. "Prompt injection" (instruction override), "jailbreak" (persona and
   roleplay attacks that defeat the safety policy) and "harmful content"
   (requests for dangerous information) are three distinct problems. A single
   pooled score hides which one a system is bad at. Every row here carries an
   `attack_class`, so metrics can be reported per class.

2. SINGLE SOURCE. One dataset measures one distribution and will not generalise.
   This pulls from every reachable public source and records provenance per row.

3. LANGUAGE. The original dataset is largely German while the configured encoder
   is English-only, which was the dominant cause of missed detections. Every row
   is language-tagged so that effect is measurable rather than confounding.

DESIGN NOTES
------------
- Sources that are gated, removed, or unreachable are SKIPPED, not fatal. Every
  skip is recorded in the manifest with its reason, so the suite is always
  reproducible and never silently smaller than it claims to be.
- Rows are deduplicated on normalised text across ALL sources. Several public
  injection datasets are forks of each other (JasperLS/prompt-injections is a
  copy of deepset/prompt-injections); without dedup the overlap would be
  double-counted and inflate whichever class it belongs to.
- Per-source caps keep the suite runnable. Sampling is seeded and recorded.

Usage:
    python -m scripts.build_eval_suite
    python -m scripts.build_eval_suite --benign-cap 2000
"""
import argparse
import hashlib
import json
import os
import random
import re
import unicodedata

OUTPUT_FILE = os.path.join("data", "eval_suite.jsonl")
MANIFEST_FILE = os.path.join("_evidence", "eval_suite_manifest.json")
SEED = 20260723

# Attack taxonomy. `benign` is not an attack; it is the negative class.
CLASS_BENIGN = "benign"
CLASS_INJECTION = "prompt_injection"
CLASS_JAILBREAK = "jailbreak"
CLASS_HARMFUL = "harmful_content"


# ---------------------------------------------------------------------------
# Language identification (de/en discrimination, dependency-free)
# ---------------------------------------------------------------------------

_DE_MARKERS = {
    "der", "die", "das", "und", "ist", "nicht", "ich", "du", "sie", "wir", "ein",
    "eine", "einen", "einem", "einer", "mit", "auf", "fuer", "für", "von", "zu",
    "dass", "wie", "was", "wenn", "aber", "auch", "sind", "haben", "werden",
    "kann", "soll", "muss", "alle", "alles", "jetzt", "bitte", "vergiss",
    "ignoriere", "schreibe", "deine", "deinen", "mich", "mir", "es", "im", "am",
}
_EN_MARKERS = {
    "the", "and", "is", "not", "you", "we", "a", "an", "with", "on", "for",
    "from", "to", "that", "how", "what", "if", "but", "also", "are", "have",
    "will", "can", "should", "must", "all", "now", "please", "forget", "ignore",
    "write", "your", "me", "my", "it", "in", "of", "this", "do",
}


def detect_language(text: str) -> str:
    """
    Coarse language tag: 'de', 'en', or 'other'.

    A stopword-overlap heuristic rather than a dependency. This only needs to
    separate German from English well enough to stratify metrics; it is not a
    general-purpose language identifier and is labelled as such in the manifest.
    """
    tokens = set(re.findall(r"[a-zA-ZäöüÄÖÜß]+", text.lower()))
    if not tokens:
        return "other"
    de = len(tokens & _DE_MARKERS)
    en = len(tokens & _EN_MARKERS)
    # Characters that essentially only appear in German among these sources.
    if any(c in text for c in "äöüßÄÖÜ"):
        de += 2
    if de == 0 and en == 0:
        return "other"
    if de > en:
        return "de"
    if en > de:
        return "en"
    return "other"


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalisation so near-identical rows across forks collapse."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text)).strip()


# ---------------------------------------------------------------------------
# Source adapters. Each yields (text, label, attack_class).
# label: 1 = attack, 0 = benign.
# ---------------------------------------------------------------------------

def _src_deepset(load):
    """Instruction-override injections, German-heavy. The original dataset."""
    ds = load("deepset/prompt-injections", split="train")
    for row in ds:
        label = int(row["label"])
        yield row["text"], label, (CLASS_INJECTION if label else CLASS_BENIGN)


def _src_jailbreak_classification(load):
    """Persona/roleplay jailbreaks (DAN-style) with a matched benign set."""
    for split in ("train", "test"):
        ds = load("jackhhao/jailbreak-classification", split=split)
        for row in ds:
            is_jb = str(row["type"]).strip().lower() == "jailbreak"
            yield row["prompt"], int(is_jb), (CLASS_JAILBREAK if is_jb else CLASS_BENIGN)


def _src_gandalf(load):
    """Lakera Gandalf: real attacker attempts to extract a secret. All attacks."""
    ds = load("lakera/gandalf_ignore_instructions", split="train")
    for row in ds:
        yield row["text"], 1, CLASS_INJECTION


def _src_chatgpt_jailbreaks(load):
    """Curated in-the-wild jailbreak prompts. All attacks."""
    ds = load("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
    for row in ds:
        text = (row.get("Prompt") or "").strip()
        if text:
            yield text, 1, CLASS_JAILBREAK


def _src_toxic_chat(load):
    """
    Real user traffic with human toxicity + jailbreaking annotations.
    Valuable because its benign portion is genuine production-like traffic,
    which is exactly what a false-positive rate should be measured against.
    """
    ds = load("lmsys/toxic-chat", "toxicchat0124", split="train")
    for row in ds:
        text = (row.get("user_input") or "").strip()
        if not text:
            continue
        if int(row.get("jailbreaking", 0) or 0) == 1:
            yield text, 1, CLASS_JAILBREAK
        elif int(row.get("toxicity", 0) or 0) == 1:
            yield text, 1, CLASS_HARMFUL
        else:
            yield text, 0, CLASS_BENIGN


def _src_jbb_behaviors(load):
    """JailbreakBench harmful/benign behaviour pairs."""
    for split, label, cls in (("harmful", 1, CLASS_HARMFUL), ("benign", 0, CLASS_BENIGN)):
        ds = load("JailbreakBench/JBB-Behaviors", "behaviors", split=split)
        for row in ds:
            text = (row.get("Goal") or row.get("goal") or "").strip()
            if text:
                yield text, label, cls


def _src_alpaca(load):
    """
    General benign instructions. Included because every attack-focused dataset
    has an unrealistically adversarial benign class; a guardrail's FPR must be
    measured against ordinary traffic, not against near-miss attacks only.
    """
    ds = load("tatsu-lab/alpaca", split="train")
    for row in ds:
        text = (row.get("instruction") or "").strip()
        if text:
            yield text, 0, CLASS_BENIGN


SOURCES = [
    {"name": "deepset/prompt-injections", "fn": _src_deepset, "cap": None,
     "note": "Original project benchmark. German-heavy instruction override."},
    {"name": "jackhhao/jailbreak-classification", "fn": _src_jailbreak_classification,
     "cap": None, "note": "DAN-style persona jailbreaks + benign."},
    {"name": "lakera/gandalf_ignore_instructions", "fn": _src_gandalf, "cap": 800,
     "note": "Real attacker attempts from the Gandalf game. Attacks only."},
    {"name": "rubend18/ChatGPT-Jailbreak-Prompts", "fn": _src_chatgpt_jailbreaks,
     "cap": None, "note": "In-the-wild jailbreaks. Attacks only."},
    {"name": "lmsys/toxic-chat", "fn": _src_toxic_chat, "cap": 3000,
     "note": "Real user traffic, human-annotated. Realistic benign distribution."},
    {"name": "JailbreakBench/JBB-Behaviors", "fn": _src_jbb_behaviors, "cap": None,
     "note": "Harmful-behaviour goals paired with benign controls."},
    {"name": "tatsu-lab/alpaca", "fn": _src_alpaca, "cap": 1200,
     "note": "Ordinary instruction traffic. Anchors the false-positive rate."},
]

# Sources deliberately not included, recorded so the omission is explicit.
EXCLUDED = {
    "hackaprompt/hackaprompt-dataset": "gated on the Hub; requires authentication",
    "walledai/HarmBench": "gated on the Hub; requires authentication",
    "JasperLS/prompt-injections": "byte-identical fork of deepset/prompt-injections; "
                                  "would be fully removed by dedup anyway",
}


def build(benign_cap=None):
    from datasets import load_dataset

    def load(name, *args, **kwargs):
        return load_dataset(name, *args, **kwargs)

    rng = random.Random(SEED)
    records = []
    seen = {}
    provenance = []
    duplicates = 0

    for spec in SOURCES:
        name = spec["name"]
        print(f"\n[{name}] loading...")
        try:
            rows = list(spec["fn"](load))
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  SKIPPED - {reason}")
            provenance.append({
                "source": name, "status": "skipped", "reason": reason,
                "kept": 0, "note": spec["note"],
            })
            continue

        raw_n = len(rows)
        cap = spec["cap"]
        if cap and raw_n > cap:
            rng.shuffle(rows)
            rows = rows[:cap]

        kept = 0
        dup_here = 0
        for text, label, cls in rows:
            text = (text or "").strip()
            if not text or len(text) < 3:
                continue
            key = normalize_for_dedup(text)
            if not key:
                continue
            if key in seen:
                dup_here += 1
                duplicates += 1
                continue
            seen[key] = name
            records.append({
                "id": hashlib.sha1(key.encode("utf-8")).hexdigest()[:16],
                "text": text,
                "label": label,
                "attack_class": cls,
                "source": name,
                "language": detect_language(text),
            })
            kept += 1

        print(f"  raw={raw_n} sampled={len(rows)} kept={kept} duplicates_dropped={dup_here}")
        provenance.append({
            "source": name, "status": "ok", "raw_rows": raw_n,
            "sampled": len(rows), "kept": kept, "duplicates_dropped": dup_here,
            "cap": cap, "note": spec["note"],
        })

    # Optional global benign cap, applied last so it does not bias any one source.
    if benign_cap:
        benign = [r for r in records if r["label"] == 0]
        attacks = [r for r in records if r["label"] == 1]
        if len(benign) > benign_cap:
            rng.shuffle(benign)
            benign = benign[:benign_cap]
        records = attacks + benign

    rng.shuffle(records)
    return records, provenance, duplicates


def summarize(records, provenance, duplicates, benign_cap):
    def tally(key):
        out = {}
        for r in records:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    cross = {}
    for r in records:
        cross.setdefault(r["attack_class"], {})
        cross[r["attack_class"]][r["language"]] = \
            cross[r["attack_class"]].get(r["language"], 0) + 1

    n_attack = sum(1 for r in records if r["label"] == 1)
    return {
        "seed": SEED,
        "total": len(records),
        "attacks": n_attack,
        "benign": len(records) - n_attack,
        "duplicates_dropped": duplicates,
        "benign_cap": benign_cap,
        "by_attack_class": tally("attack_class"),
        "by_language": tally("language"),
        "by_source": tally("source"),
        "class_by_language": cross,
        "sources": provenance,
        "excluded_sources": EXCLUDED,
        "caveats": [
            "Language tags come from a stopword-overlap heuristic (de/en/other), "
            "not a trained language identifier. Adequate for stratifying metrics; "
            "do not cite as ground-truth language labels.",
            "Attack-class labels are assigned per SOURCE, inheriting each "
            "dataset's own labelling conventions. They are not independently "
            "re-annotated, so class boundaries between jailbreak and "
            "harmful_content are approximate where a source mixes them.",
            "Benign rows are pooled from adversarial-adjacent datasets AND from "
            "ordinary instruction traffic (alpaca, toxic-chat). Report FPR "
            "separately for each if a realistic production estimate is needed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Build the evaluation suite")
    parser.add_argument("--benign-cap", type=int, default=None,
                        help="Cap total benign rows after pooling.")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("_evidence", exist_ok=True)

    records, provenance, duplicates = build(benign_cap=args.benign_cap)
    manifest = summarize(records, provenance, duplicates, args.benign_cap)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation suite built ===")
    print(f"  total   : {manifest['total']}")
    print(f"  attacks : {manifest['attacks']}")
    print(f"  benign  : {manifest['benign']}")
    print(f"  dupes   : {manifest['duplicates_dropped']} dropped")
    print(f"\n  by attack class : {manifest['by_attack_class']}")
    print(f"  by language     : {manifest['by_language']}")
    skipped = [p['source'] for p in provenance if p['status'] == 'skipped']
    if skipped:
        print(f"\n  SKIPPED SOURCES : {skipped}")
    print(f"\n  suite    -> {OUTPUT_FILE}")
    print(f"  manifest -> {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
