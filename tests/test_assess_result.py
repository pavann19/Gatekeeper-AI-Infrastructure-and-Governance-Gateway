"""AssessResult.to_dict() must reproduce the `details` dict that
core.risk.assess_risk returned as inline literals, byte-for-byte, for every
one of the 5 return paths — same keys, same order.
"""
import pytest

from core.assess_result import AssessResult, _KEYS


def test_cache_locked_high_shape():
    want = {"semantic_score": 0.91, "source": "cache_locked_high",
            "educational_context": False, "domain_score": None, "topicality": "UNKNOWN",
            "symbolic_triggered": False, "judge_invoked": False, "meta_intent_score": None,
            "fusion_triggering_class": None, "fusion_class_scores": {}}
    r = AssessResult.cache_hit("HIGH", 0.91, locked_high=True)
    assert r.to_dict() == want
    assert list(r.to_dict()) == list(want)
    assert r.risk_level == "HIGH"


def test_cache_shape():
    want = {"semantic_score": 0.42, "source": "cache",
            "educational_context": False, "domain_score": None, "topicality": "UNKNOWN",
            "symbolic_triggered": False, "judge_invoked": False, "meta_intent_score": None,
            "fusion_triggering_class": None, "fusion_class_scores": {}}
    r = AssessResult.cache_hit("MEDIUM", 0.42)
    assert r.to_dict() == want
    assert list(r.to_dict()) == list(want)
    assert r.risk_level == "MEDIUM"


def test_symbolic_shape():
    want = {"source": "symbolic_rule", "detail": "INSTRUCTION_OVERRIDE_DETECTED",
            "semantic_score": 1.0, "educational_context": False, "domain_score": None,
            "topicality": "UNKNOWN", "symbolic_triggered": True, "judge_invoked": False,
            "meta_intent_score": None, "fusion_triggering_class": None,
            "fusion_class_scores": {}}
    r = AssessResult.symbolic("INSTRUCTION_OVERRIDE_DETECTED")
    assert r.to_dict() == want
    assert list(r.to_dict()) == list(want)
    assert r.risk_level == "HIGH"


def test_fast_path_shape():
    want = {"source": "fast_path_meta_intent", "semantic_score": 1.0,
            "educational_context": False, "domain_score": None, "topicality": "UNKNOWN",
            "symbolic_triggered": False, "judge_invoked": False, "meta_intent_score": 0.83,
            "fusion_triggering_class": None, "fusion_class_scores": {}}
    r = AssessResult.fast_path("HIGH", "fast_path_meta_intent", 0.83)
    assert r.to_dict() == want
    assert list(r.to_dict()) == list(want)


def test_deep_path_shape():
    want = {"semantic_score": 0.55, "source": "fusion_clean_pass",
            "educational_context": False, "domain_score": None, "topicality": "ON_TOPIC",
            "symbolic_triggered": False, "judge_invoked": False, "centroid_score": 0.3,
            "fusion_available": True, "fusion_detail": "fusion applied",
            "fusion_detector_scores": {"anchors": 0.1}, "fusion_triggering_class": None,
            "fusion_class_scores": {}, "anchor_threat_score": 0.12,
            "meta_intent_score": 0.05, "meta_intent_ms": 4.2,
            "faiss_threat_search_ms": 1.1, "domain_alignment_ms": 0.0, "fusion_ms": 37.5}
    r = AssessResult.deep_path(
        "LOW", "fusion_clean_pass",
        semantic_score=0.55, is_educational=False, domain_score=None, topicality="ON_TOPIC",
        judge_invoked=False, centroid_score=0.3, fusion_available=True,
        fusion_detail="fusion applied", fusion_detector_scores={"anchors": 0.1},
        fusion_triggering_class=None, fusion_class_scores={}, anchor_threat_score=0.12,
        meta_intent_score=0.05, meta_intent_ms=4.2, faiss_threat_search_ms=1.1,
        domain_alignment_ms=0.0, fusion_ms=37.5,
    )
    assert r.to_dict() == want
    assert list(r.to_dict()) == list(want)


def test_deep_path_missing_timing_is_type_error():
    """The Finding-C guard: omitting a deep-path field (here fusion_ms) must
    fail loudly at construction, not silently drop a metric."""
    with pytest.raises(TypeError):
        AssessResult.deep_path(
            "LOW", "x", semantic_score=0, is_educational=False, domain_score=None,
            topicality="X", judge_invoked=False, centroid_score=0, fusion_available=True,
            fusion_detail="", fusion_detector_scores={}, fusion_triggering_class=None,
            fusion_class_scores={}, anchor_threat_score=0, meta_intent_score=0,
            meta_intent_ms=1, faiss_threat_search_ms=1, domain_alignment_ms=1,
            # fusion_ms deliberately omitted
        )


def test_context_keys_are_exhaustive():
    """Every _KEYS context maps to attributes that actually exist on the dataclass."""
    fields = set(AssessResult.__dataclass_fields__)
    for ctx, keys in _KEYS.items():
        assert set(keys) <= fields, f"{ctx}: unknown keys {set(keys) - fields}"
