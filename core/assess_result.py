"""Typed replacement for the 5 hand-built `details` dicts returned by
`core.risk.assess_risk`.

WHY: `assess_risk` returns `(risk_level, details)` from 5 code paths, each
assembling `details` as a raw literal. In Aug the deep-path literal stopped
copying the per-stage `*_ms` timings from `signals` — they were computed and
silently dropped, leaving the Prometheus stage histogram empty in production
(fixed in 89e8dcc, but by patching the literal, not the class of bug).

Each construction context is a classmethod whose signature lists exactly the
fields that context produces. Forgetting one is a TypeError at construction,
not a silent omission. `to_dict()` reproduces the wire shape per context
byte-for-byte (see tests/test_assess_result.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Key order per context — matches the literals previously inlined in core/risk.py.
_KEYS = {
    "cache": ("semantic_score", "source", "educational_context", "domain_score",
              "topicality", "symbolic_triggered", "judge_invoked",
              "meta_intent_score", "fusion_triggering_class", "fusion_class_scores"),
    "cache_locked_high": ("semantic_score", "source", "educational_context",
                          "domain_score", "topicality", "symbolic_triggered",
                          "judge_invoked", "meta_intent_score",
                          "fusion_triggering_class", "fusion_class_scores"),
    "symbolic": ("source", "detail", "semantic_score", "educational_context",
                 "domain_score", "topicality", "symbolic_triggered",
                 "judge_invoked", "meta_intent_score", "fusion_triggering_class",
                 "fusion_class_scores"),
    "fast_path": ("source", "semantic_score", "educational_context", "domain_score",
                  "topicality", "symbolic_triggered", "judge_invoked",
                  "meta_intent_score", "fusion_triggering_class", "fusion_class_scores"),
    "deep_path": ("semantic_score", "source", "educational_context", "domain_score",
                  "topicality", "symbolic_triggered", "judge_invoked",
                  "centroid_score", "fusion_available", "fusion_detail",
                  "fusion_detector_scores", "fusion_triggering_class",
                  "fusion_class_scores", "anchor_threat_score", "meta_intent_score",
                  "meta_intent_ms", "faiss_threat_search_ms", "domain_alignment_ms",
                  "fusion_ms"),
}


@dataclass
class AssessResult:
    risk_level: str
    source: str
    _context: str = "deep_path"
    semantic_score: float | None = None
    detail: str | None = None
    educational_context: bool = False
    domain_score: float | None = None
    topicality: str = "UNKNOWN"
    symbolic_triggered: bool = False
    judge_invoked: bool = False
    meta_intent_score: float | None = None
    fusion_triggering_class: str | None = None
    fusion_class_scores: dict = field(default_factory=dict)
    centroid_score: float | None = None
    fusion_available: bool | None = None
    fusion_detail: str | None = None
    fusion_detector_scores: dict | None = None
    anchor_threat_score: float | None = None
    meta_intent_ms: float | None = None
    faiss_threat_search_ms: float | None = None
    domain_alignment_ms: float | None = None
    fusion_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """The wire `details` dict for this result's context — same keys, same
        order as the literals previously inlined in core.risk.assess_risk."""
        return {k: getattr(self, k) for k in _KEYS[self._context]}

    def as_return(self) -> tuple[str, dict[str, Any]]:
        return self.risk_level, self.to_dict()

    # --- one constructor per assess_risk return path ---------------------

    @classmethod
    def cache_hit(cls, risk_level: str, semantic_score: float, *, locked_high: bool = False):
        return cls(risk_level=risk_level,
                   source="cache_locked_high" if locked_high else "cache",
                   _context="cache_locked_high" if locked_high else "cache",
                   semantic_score=semantic_score)

    @classmethod
    def symbolic(cls, detail: str):
        return cls(risk_level="HIGH", source="symbolic_rule", _context="symbolic",
                   detail=detail, semantic_score=1.0, symbolic_triggered=True)

    @classmethod
    def fast_path(cls, risk_level: str, source: str, meta_intent_score: float):
        return cls(risk_level=risk_level, source=source, _context="fast_path",
                   semantic_score=1.0, meta_intent_score=meta_intent_score)

    @classmethod
    def deep_path(cls, risk_level: str, source: str, *, semantic_score, is_educational,
                  domain_score, topicality, judge_invoked, centroid_score,
                  fusion_available, fusion_detail, fusion_detector_scores,
                  fusion_triggering_class, fusion_class_scores, anchor_threat_score,
                  meta_intent_score, meta_intent_ms, faiss_threat_search_ms,
                  domain_alignment_ms, fusion_ms):
        # Every deep-path field is a REQUIRED keyword here — omitting one
        # (the Finding-C bug) is a TypeError, not a dropped metric.
        return cls(
            risk_level=risk_level, source=source, _context="deep_path",
            semantic_score=semantic_score, educational_context=is_educational,
            domain_score=domain_score, topicality=topicality,
            judge_invoked=judge_invoked, centroid_score=centroid_score,
            fusion_available=fusion_available, fusion_detail=fusion_detail,
            fusion_detector_scores=fusion_detector_scores,
            fusion_triggering_class=fusion_triggering_class,
            fusion_class_scores=fusion_class_scores or {},
            anchor_threat_score=anchor_threat_score,
            meta_intent_score=meta_intent_score, meta_intent_ms=meta_intent_ms,
            faiss_threat_search_ms=faiss_threat_search_ms,
            domain_alignment_ms=domain_alignment_ms, fusion_ms=fusion_ms,
        )
