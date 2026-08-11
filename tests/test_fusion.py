"""
Unit tests for Stage 3 (deterministic fusion).

These exercise the decision logic with no models in the loop — the whole point
of separating signal collection from fusion is that the decision boundary is
testable in isolation.
"""
import pytest

from core import risk as risk_mod
from core.risk import classify_topicality, fuse_signals


def make_signals(**overrides):
    """A benign, in-domain, no-threat baseline signal set."""
    signals = {
        "meta_intent_score": 0.0,
        "threat_score": 0.0,
        "dynamic_threat_score": 0.0,
        "is_educational": False,
        "domain_aligned": True,
        "domain_score": 0.9,
        "centroid_score": 0.0,
    }
    signals.update(overrides)
    return signals


@pytest.fixture
def domain_mode(monkeypatch):
    """Sets DOMAIN_GUARDRAIL_MODE for the duration of a test."""
    def _set(mode):
        monkeypatch.setattr(risk_mod.settings, "DOMAIN_GUARDRAIL_MODE", mode)
    return _set


# --- topicality classification ---------------------------------------------

def test_topicality_unknown_when_not_evaluated():
    assert classify_topicality(make_signals(domain_aligned=None)) == "UNKNOWN"


def test_topicality_in_domain():
    assert classify_topicality(make_signals(domain_aligned=True)) == "IN_DOMAIN"


def test_topicality_out_of_domain():
    assert classify_topicality(make_signals(domain_aligned=False)) == "OUT_OF_DOMAIN"


# --- the regression this whole fix exists for ------------------------------

def test_off_topic_benign_prompt_is_not_a_safety_risk(domain_mode):
    """
    THE core regression test.

    An off-domain prompt with zero threat signal must come back LOW. The
    pre-fix pipeline returned MEDIUM here, and because the benchmark scored
    MEDIUM as a malice prediction, 79 of 80 benign prompts were recorded as
    false positives (FPR ~98%).
    """
    domain_mode("advisory")
    signals = make_signals(domain_aligned=False, domain_score=0.05)

    level, source, judge_required, topicality = fuse_signals(signals, "How do I bake bread?")

    assert level == "LOW"
    assert source == "clean_pass"
    assert topicality == "OUT_OF_DOMAIN"
    assert judge_required is False


def test_off_topic_escalates_only_in_enforcing_mode(domain_mode):
    domain_mode("enforcing")
    signals = make_signals(domain_aligned=False, domain_score=0.05)

    level, source, _, topicality = fuse_signals(signals, "How do I bake bread?")

    assert level == "MEDIUM"
    assert source == "domain_guardrail"
    assert topicality == "OUT_OF_DOMAIN"


def test_domain_guardrail_default_is_off():
    """A third-party deployment must not inherit this repo's domain corpus."""
    from core.config import Settings
    assert Settings().DOMAIN_GUARDRAIL_MODE == "off"


def test_invalid_domain_mode_rejected():
    from core.config import Settings
    with pytest.raises(ValueError):
        Settings(DOMAIN_GUARDRAIL_MODE="sometimes")


# --- safety signals still fire ---------------------------------------------

def test_meta_intent_vetoes_everything(domain_mode):
    domain_mode("off")
    signals = make_signals(meta_intent_score=0.95, domain_aligned=None)

    level, source, judge_required, _ = fuse_signals(signals, "...")

    assert level == "HIGH"
    assert source == "semantic_meta_intent"
    assert judge_required is False


def test_meta_intent_outranks_domain_guardrail(domain_mode):
    """Safety verdicts must not be masked by a scoping verdict."""
    domain_mode("enforcing")
    signals = make_signals(meta_intent_score=0.95, domain_aligned=False, domain_score=0.01)

    level, source, _, _ = fuse_signals(signals, "...")

    assert level == "HIGH"
    assert source == "semantic_meta_intent"


def test_high_threat_score_blocks(domain_mode):
    domain_mode("off")
    signals = make_signals(threat_score=0.99, domain_aligned=None)

    level, source, judge_required, _ = fuse_signals(signals, "...")

    assert level == "HIGH"
    assert source == "vector_threat_critical"
    assert judge_required is False


def test_ambiguous_threat_requests_judge(domain_mode):
    domain_mode("off")
    signals = make_signals(threat_score=0.30, domain_aligned=None)

    level, source, judge_required, _ = fuse_signals(signals, "...")

    assert level == "MEDIUM"
    assert source == "judge_pending"
    assert judge_required is True


def test_educational_context_short_circuits_judge(domain_mode):
    domain_mode("off")
    signals = make_signals(threat_score=0.30, is_educational=True, domain_aligned=None)

    level, source, judge_required, _ = fuse_signals(signals, "...")

    assert level == "MEDIUM"
    assert source == "educational_safe_harbor"
    assert judge_required is False


def test_clean_prompt_passes(domain_mode):
    domain_mode("off")
    signals = make_signals(domain_aligned=None)

    level, source, judge_required, topicality = fuse_signals(signals, "What is a hash map?")

    assert level == "LOW"
    assert source == "clean_pass"
    assert judge_required is False
    assert topicality == "UNKNOWN"


# --- multi-detector fusion branch -------------------------------------------
#
# core/fusion.py adds four keys to the signals dict: fusion_available,
# fusion_score, fusion_threshold_high, fusion_threshold_medium. These tests
# pin the CONTRACT between core/fusion.py and this Stage-3 cascade: when
# fusion ran, its score and thresholds drive the decision and the source is
# labelled "fusion_*"; when it did not (fusion_available absent or False),
# behaviour is byte-for-byte the pre-fusion anchors-only path exercised above.

def make_fusion_signals(**overrides):
    signals = make_signals(threat_score=0.0)
    signals.update({
        "fusion_available": True,
        "fusion_score": 0.0,
        "fusion_threshold_high": 0.28,
        "fusion_threshold_medium": 0.10,
    })
    signals.update(overrides)
    return signals


def test_fusion_high_risk_uses_fusion_threshold_not_legacy_threshold(domain_mode):
    """
    A score that would be LOW under the legacy anchors-only threshold (0.48)
    but is ABOVE the fusion policy's calibrated HIGH threshold (0.28) must
    still block — proving the fusion threshold, not the legacy constant, is
    what gates the decision when fusion is available.
    """
    domain_mode("off")
    signals = make_fusion_signals(fusion_score=0.30, threat_score=0.0)

    level, source, judge_required, _ = fuse_signals(signals, "prompt")

    assert level == "HIGH"
    assert source == "fusion_threat_critical"
    assert judge_required is False


def test_fusion_medium_zone_routes_to_judge(domain_mode):
    domain_mode("off")
    signals = make_fusion_signals(fusion_score=0.15, is_educational=False)

    level, source, judge_required, _ = fuse_signals(signals, "prompt")

    assert level == "MEDIUM"
    assert source == "fusion_judge_pending"
    assert judge_required is True


def test_fusion_medium_zone_educational_safe_harbor(domain_mode):
    domain_mode("off")
    signals = make_fusion_signals(fusion_score=0.15, is_educational=True)

    level, source, judge_required, _ = fuse_signals(signals, "prompt")

    assert level == "MEDIUM"
    assert source == "fusion_educational_safe_harbor"
    assert judge_required is False


def test_fusion_below_medium_threshold_clean_pass(domain_mode):
    domain_mode("off")
    signals = make_fusion_signals(fusion_score=0.02)

    level, source, judge_required, _ = fuse_signals(signals, "prompt")

    assert level == "LOW"
    assert source == "fusion_clean_pass"
    assert judge_required is False


def test_fusion_unavailable_falls_back_to_legacy_anchors_path(domain_mode):
    """
    THE FALLBACK CONTRACT. When fusion_available is False, behaviour must be
    IDENTICAL to the pre-fusion cascade — same thresholds, same source labels
    — regardless of whatever garbage sits in fusion_score. A caller that
    forgets to check availability must not get a decision driven by a stale
    or default fusion_score.
    """
    domain_mode("off")
    signals = make_fusion_signals(
        fusion_available=False,
        fusion_score=0.99,       # would be HIGH if this were consulted
        threat_score=0.50,       # above legacy SEMANTIC_THRESHOLD_HIGH (0.48)
    )

    level, source, judge_required, _ = fuse_signals(signals, "prompt")

    assert level == "HIGH"
    assert source == "vector_threat_critical"  # legacy label, not "fusion_*"


def test_fusion_key_absent_also_falls_back(domain_mode):
    """Signals dicts that never learned about fusion (e.g. older callers) must
    behave exactly as before — this is what keeps every pre-existing test in
    this file passing unchanged."""
    domain_mode("off")
    signals = make_signals(threat_score=0.50)
    assert "fusion_available" not in signals

    level, source, judge_required, _ = fuse_signals(signals, "prompt")

    assert level == "HIGH"
    assert source == "vector_threat_critical"


# --- judge arbitration ------------------------------------------------------

@pytest.mark.parametrize(
    "verdict,threat_present,expected_risk,expected_source",
    [
        ("DANGEROUS", False, "HIGH", "semantic_judge"),
        # By DEFAULT the arbiter has no clear-authority: a SAFE verdict caps at
        # MEDIUM even in the lower band. Measured — the judge spent that
        # authority to clear 13 attacks and only 4 benign prompts.
        # See core/config.py::JUDGE_MAY_CLEAR_TO_LOW.
        ("SAFE", False, "MEDIUM", "semantic_judge_override_capped"),
        # A SAFE verdict must not fully clear a prompt that tripped a threat signal.
        ("SAFE", True, "MEDIUM", "semantic_judge_override_restricted"),
        ("AMBIGUOUS", False, "MEDIUM", "semantic_judge_ambiguous"),
        # Fail-closed on every abnormal verdict.
        ("JUDGE_OFFLINE", False, "HIGH", "judge_failure_fail_closed"),
        ("garbage", False, "HIGH", "judge_failure_fail_closed"),
    ],
)
def test_judge_arbitration(monkeypatch, verdict, threat_present, expected_risk, expected_source):
    monkeypatch.setattr(risk_mod, "semantic_judge", lambda p: verdict)

    level, source = risk_mod.judge_arbitration("...", threat_present=threat_present)

    assert level == expected_risk
    assert source == expected_source
