"""
Session-wide test fixtures.

autouse fixtures here apply to EVERY test in the suite, not just files that
explicitly request them — used sparingly, only for global state that would
otherwise leak between tests and cause order-dependent failures.
"""
import pytest

from core.circuit_breaker import llama_guard_breaker, ollama_judge_breaker


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """
    core/circuit_breaker.py's breakers are module-level singletons shared by
    every call in the process — deliberately, so they track a backend's real
    health across requests. That same property means failures recorded by one
    test (e.g. a test that feeds llama_guard_arbitration a raising stub) would
    otherwise accumulate toward the failure_threshold and could trip the
    breaker for a LATER, unrelated test that expects a call to actually be
    attempted. Reset before and after every test so each starts from a known
    closed state, independent of what ran before it.
    """
    ollama_judge_breaker.reset()
    llama_guard_breaker.reset()
    yield
    ollama_judge_breaker.reset()
    llama_guard_breaker.reset()
