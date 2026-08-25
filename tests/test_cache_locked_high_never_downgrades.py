"""
Regression coverage for a real gap found by manual mutation testing
(Phase 8 SDLC hardening pass): core/risk.py's `assess_risk` has a comment
declaring "SAFETY: Never downgrade a HIGH-risk cached decision" guarding a
`if cached_risk == "HIGH":` branch -- but until this file, NOTHING in the
test suite actually asserted `risk == "HIGH"` for that path. The existing
coverage (tests/test_per_class_risk_vector.py's `cache_locked_high`
parametrization) only checked that `fusion_class_scores`/
`fusion_triggering_class` are present-but-empty on early-return paths; it
never checked the risk verdict itself.

Proof this was a real gap, not a hypothetical one: inverting the
comparison to `if cached_risk == "NEVER_MATCHES_XYZ_SENTINEL":` (a mutant
that completely disables the guard, so a cached HIGH would fall through to
the generic non-locked cache-hit branch) left the FULL existing test suite
green -- the mutant survived. These tests kill it.
"""
from unittest.mock import patch

from core.risk import assess_risk


def test_cached_high_risk_returns_high_via_the_locked_path():
    with patch("core.risk._ensure_faiss_initialized"), \
         patch("core.risk.get_embedding", return_value=[0.0]), \
         patch("core.risk.lookup_cache", return_value=("HIGH", 0.97)):
        risk, details = assess_risk("irrelevant, cache answers first")

    assert risk == "HIGH"
    assert details["source"] == "cache_locked_high"


def test_cached_medium_risk_is_not_forced_through_the_locked_path():
    """The locked path is HIGH-specific -- a cached MEDIUM/LOW must take
    the ordinary (still early-return, but not "locked") cache-hit branch,
    reporting its own real cached value, not "HIGH"."""
    with patch("core.risk._ensure_faiss_initialized"), \
         patch("core.risk.get_embedding", return_value=[0.0]), \
         patch("core.risk.lookup_cache", return_value=("MEDIUM", 0.55)):
        risk, details = assess_risk("irrelevant, cache answers first")

    assert risk == "MEDIUM"
    assert details["source"] == "cache"


def test_cached_low_risk_is_not_forced_through_the_locked_path():
    with patch("core.risk._ensure_faiss_initialized"), \
         patch("core.risk.get_embedding", return_value=[0.0]), \
         patch("core.risk.lookup_cache", return_value=("LOW", 0.02)):
        risk, details = assess_risk("irrelevant, cache answers first")

    assert risk == "LOW"
    assert details["source"] == "cache"
