"""
Regression coverage for assess_risk's `raw_prompt` parameter (Phase 8 live
pentest fix): the local, in-process text classifiers (hard-ban regex,
domain alignment, the fusion sub-detectors) run on `raw_prompt` when
supplied, while the embedding, the persisted semantic cache, and the judge
escalation all stay on `prompt` (the redacted text) exactly as before.

Root cause this fixes: `[REDACTED:PERSON]`-style placeholder tokens are
themselves out-of-distribution for classifiers trained on natural language,
so a benign redacted prompt could get falsely flagged. See
core/risk.py::assess_risk's docstring and core/config.py's
HALLUCINATION_CHECK_ENABLED docstring for the full story (a second, related
finding from the same pentest).

Every dependency is mocked at the module level so this tests ONLY the
routing of raw vs. redacted text through assess_risk, not the real ML
models' behaviour (already covered elsewhere).
"""
from unittest.mock import patch

from core.risk import assess_risk


def _patch_pipeline(**overrides):
    """Common no-op-ish mocks for every assess_risk dependency, so a test
    only has to override the ones it cares about."""
    defaults = dict(
        get_embedding=lambda text: [0.0],
        lookup_cache=lambda prompt, vec: (None, None),
        hard_ban_triggered=lambda text: (False, None),
        _fast_path_signals=lambda vec: {"meta_intent_score": 0.0, "threat_score": 0.0},
        _fast_path_decision=lambda fast: None,
        collect_semantic_signals=lambda text, vec, fast=None: {
            "is_educational": False, "domain_score": None, "centroid_score": 0.0,
            "fusion_available": False, "fusion_score": 0.0, "fusion_detail": "n/a",
            "fusion_detector_scores": {}, "fusion_triggering_class": None,
            "fusion_class_scores": {}, "threat_score": 0.0, "meta_intent_score": 0.0,
        },
        fuse_signals=lambda signals, text: ("LOW", "test_source", False, "UNKNOWN"),
        save_cache_entry=lambda *a, **k: None,
        _ensure_faiss_initialized=lambda: None,
    )
    defaults.update(overrides)
    return defaults


def test_raw_prompt_routed_to_hard_ban_not_redacted_prompt():
    seen = {}

    def _capture_hard_ban(text):
        seen["hard_ban_text"] = text
        return False, None

    patches = _patch_pipeline(hard_ban_triggered=_capture_hard_ban)
    with patch.multiple("core.risk", **patches):
        assess_risk("REDACTED VERSION", raw_prompt="ORIGINAL VERSION")

    assert seen["hard_ban_text"] == "ORIGINAL VERSION"


def test_raw_prompt_routed_to_collect_semantic_signals():
    seen = {}

    def _capture_signals(text, vec, fast=None):
        seen["signals_text"] = text
        return _patch_pipeline()["collect_semantic_signals"](text, vec, fast)

    patches = _patch_pipeline(collect_semantic_signals=_capture_signals)
    with patch.multiple("core.risk", **patches):
        assess_risk("REDACTED VERSION", raw_prompt="ORIGINAL VERSION")

    assert seen["signals_text"] == "ORIGINAL VERSION"


def test_raw_prompt_routed_to_fuse_signals():
    seen = {}

    def _capture_fuse(signals, text):
        seen["fuse_text"] = text
        return "LOW", "test_source", False, "UNKNOWN"

    patches = _patch_pipeline(fuse_signals=_capture_fuse)
    with patch.multiple("core.risk", **patches):
        assess_risk("REDACTED VERSION", raw_prompt="ORIGINAL VERSION")

    assert seen["fuse_text"] == "ORIGINAL VERSION"


def test_redacted_prompt_still_used_for_embedding_and_cache():
    """The privacy boundary this fix must NOT cross: the embedding (and
    therefore the persisted semantic cache) stays on the REDACTED prompt,
    never the raw one, even when raw_prompt is supplied."""
    seen = {}

    def _capture_embedding(text):
        seen["embedding_text"] = text
        return [0.0]

    def _capture_cache_save(prompt, vec, risk, score, source=None):
        seen["cache_save_prompt"] = prompt

    patches = _patch_pipeline(get_embedding=_capture_embedding, save_cache_entry=_capture_cache_save)
    with patch.multiple("core.risk", **patches):
        assess_risk("REDACTED VERSION", raw_prompt="ORIGINAL VERSION")

    assert seen["embedding_text"] == "REDACTED VERSION"
    assert seen["cache_save_prompt"] == "REDACTED VERSION"


def test_no_raw_prompt_supplied_falls_back_to_prompt_everywhere():
    """Backward compatibility: every existing caller (benchmark.py,
    eval_harness.py, any test calling assess_risk(prompt) with no third
    arg) must see IDENTICAL behaviour to before this parameter existed."""
    seen = {}

    def _capture_hard_ban(text):
        seen["hard_ban_text"] = text
        return False, None

    def _capture_signals(text, vec, fast=None):
        seen["signals_text"] = text
        return _patch_pipeline()["collect_semantic_signals"](text, vec, fast)

    def _capture_fuse(signals, text):
        seen["fuse_text"] = text
        return "LOW", "test_source", False, "UNKNOWN"

    patches = _patch_pipeline(
        hard_ban_triggered=_capture_hard_ban,
        collect_semantic_signals=_capture_signals,
        fuse_signals=_capture_fuse,
    )
    with patch.multiple("core.risk", **patches):
        assess_risk("ONLY THIS TEXT")

    assert seen["hard_ban_text"] == "ONLY THIS TEXT"
    assert seen["signals_text"] == "ONLY THIS TEXT"
    assert seen["fuse_text"] == "ONLY THIS TEXT"
