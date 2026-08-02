"""
Tests for asynchronous Stage 4 arbitration: answering the caller fast with the
Ollama judge, then verifying with Llama Guard afterwards via a background
scheduler, without adding its multi-second tail latency to the response.

The two contracts that matter, in order of importance:

1. When a background_scheduler is supplied, assess_risk must NOT block on
   Llama Guard — it answers from the fast judge and merely SCHEDULES the
   confirmation. When no scheduler is supplied (the default), behaviour is
   byte-for-byte the pre-existing synchronous path, which is what
   tests/benchmark.py depends on for ground-truth measurement.

2. llama_guard_async_confirmation is a ONE-WAY RATCHET: it may escalate a
   served decision to HIGH (by upgrading the cache) if Llama Guard disagrees
   upward, but it must NEVER downgrade one. And it must never raise into the
   caller (a background-task runner), regardless of what fails internally.
"""
import unittest.mock as mock

import pytest

from core import risk as risk_mod
from core.risk import llama_guard_async_confirmation


class StubLlamaGuard:
    def __init__(self, available=True, detail="ok", verdict="safe",
                 categories=None, raises=None):
        self._available = available
        self._detail = detail
        self._verdict = verdict
        self._categories = categories or []
        self._raises = raises

    def available(self):
        return self._available, self._detail

    def classify(self, prompt):
        if self._raises:
            raise self._raises
        return {"verdict": self._verdict, "categories": self._categories, "raw": self._verdict}


def _patch_detector(monkeypatch, stub):
    monkeypatch.setattr("core.detectors.get_detector", lambda name: stub)


@pytest.fixture
def force_judge_required(monkeypatch):
    """Same fixture pattern as test_llama_guard_arbitration.py: drives
    assess_risk to Stage 4 by mocking every earlier stage."""
    monkeypatch.setattr(risk_mod, "_ensure_faiss_initialized", lambda: None)
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: [0.0])
    monkeypatch.setattr(risk_mod, "lookup_cache", lambda prompt, vec: (None, None))
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "hard_ban_triggered", lambda p: (False, None))
    monkeypatch.setattr(risk_mod, "collect_semantic_signals", lambda p, v: {
        "threat_score": 0.35, "dynamic_threat_score": 0.0, "is_educational": False,
        "domain_score": None, "domain_aligned": None, "meta_intent_score": 0.0,
        "centroid_score": 0.0, "fusion_available": False,
    })


# --- assess_risk: async path does not block on Llama Guard ------------------

def test_async_path_answers_from_fast_judge_without_calling_llama_guard(monkeypatch, force_judge_required):
    """The core promise: with a scheduler, the response must not wait on
    Llama Guard at all."""
    monkeypatch.setattr(risk_mod, "judge_arbitration",
                        lambda prompt, threat_present=False: ("LOW", "semantic_judge_override"))
    monkeypatch.setattr(
        risk_mod, "llama_guard_arbitration",
        lambda *a, **k: pytest.fail("llama_guard_arbitration must not run synchronously in the async path"),
    )

    scheduled = []
    risk, details = risk_mod.assess_risk(
        "ambiguous prompt", background_scheduler=lambda fn, *args: scheduled.append((fn, args))
    )

    assert risk == "LOW"
    assert details["source"] == "semantic_judge_override"
    assert len(scheduled) == 1
    fn, args = scheduled[0]
    assert fn is risk_mod.llama_guard_async_confirmation
    assert args[0] == "ambiguous prompt"
    assert args[2] == "LOW" and args[3] == "semantic_judge_override"


def test_no_scheduler_keeps_the_old_synchronous_behaviour(monkeypatch, force_judge_required):
    """Backward compatibility for tests/benchmark.py and any caller that
    doesn't pass a scheduler: Llama Guard runs in-request, exactly as before
    this feature existed."""
    calls = []
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration",
                        lambda prompt, threat_present=False: calls.append("llama_guard") or ("HIGH", "llama_guard_arbitration"))
    monkeypatch.setattr(risk_mod, "judge_arbitration",
                        lambda prompt, threat_present=False: pytest.fail("must not fall back when Llama Guard succeeds"))

    risk, details = risk_mod.assess_risk("ambiguous prompt")  # no scheduler

    assert risk == "HIGH"
    assert details["source"] == "llama_guard_arbitration"
    assert calls == ["llama_guard"]


def test_scheduler_receives_the_actual_prompt_vector(monkeypatch, force_judge_required):
    """The scheduled confirmation needs the real embedding to write a
    correct cache entry if it escalates - not a placeholder."""
    sentinel_vec = object()
    monkeypatch.setattr(risk_mod, "get_embedding", lambda p: sentinel_vec)
    monkeypatch.setattr(risk_mod, "judge_arbitration",
                        lambda prompt, threat_present=False: ("LOW", "semantic_judge_override"))

    scheduled = []
    risk_mod.assess_risk("x", background_scheduler=lambda fn, *args: scheduled.append(args))

    assert scheduled[0][1] is sentinel_vec


# --- llama_guard_async_confirmation: the one-way ratchet --------------------

def test_escalates_cache_when_llama_guard_disagrees_upward(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="unsafe", categories=["S1"]))
    saved = []
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: saved.append((a, k)))

    llama_guard_async_confirmation("dangerous-looking prompt", [0.0], "LOW", "semantic_judge_override")

    assert len(saved) == 1
    args, kwargs = saved[0]
    assert args[0] == "dangerous-looking prompt"
    assert args[2] == "HIGH"
    assert kwargs.get("source") == "llama_guard_async_escalation" or args[-1] == "llama_guard_async_escalation"


def test_does_not_downgrade_when_llama_guard_is_more_lenient(monkeypatch):
    """Fast path already said HIGH; Llama Guard says safe. Must NOT touch
    the cache - a served decision is never loosened after the fact."""
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="safe"))
    saved = []
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: saved.append((a, k)))

    llama_guard_async_confirmation("prompt", [0.0], "HIGH", "vector_threat_critical")

    assert saved == []


def test_no_change_when_both_agree_on_high(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="unsafe"))
    saved = []
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: saved.append((a, k)))

    llama_guard_async_confirmation("prompt", [0.0], "HIGH", "fusion_threat_critical")

    assert saved == []  # already HIGH, nothing to escalate


def test_unavailable_llama_guard_is_a_silent_noop(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(available=False, detail="gated"))
    saved = []
    monkeypatch.setattr(risk_mod, "save_cache_entry", lambda *a, **k: saved.append((a, k)))

    llama_guard_async_confirmation("prompt", [0.0], "LOW", "semantic_judge_override")  # must not raise

    assert saved == []


def test_never_raises_even_if_llama_guard_arbitration_itself_raises(monkeypatch):
    """This runs inside a background-task runner with no caller watching for
    exceptions - it must never propagate one."""
    def boom(*a, **k):
        raise RuntimeError("cuda oom")
    monkeypatch.setattr(risk_mod, "llama_guard_arbitration", boom)

    llama_guard_async_confirmation("prompt", [0.0], "LOW", "semantic_judge_override")  # must not raise


def test_never_raises_if_the_cache_write_itself_fails(monkeypatch):
    _patch_detector(monkeypatch, StubLlamaGuard(verdict="unsafe"))
    def boom(*a, **k):
        raise IOError("disk full")
    monkeypatch.setattr(risk_mod, "save_cache_entry", boom)

    llama_guard_async_confirmation("prompt", [0.0], "LOW", "semantic_judge_override")  # must not raise
