# core/llm_providers.py
"""
Provider abstraction (Phase 5, Real LLM Gateway roadmap item — scoped down
to this piece only for this pass; request forwarding, streaming, token
accounting, retry/fallback, and audit trail are explicitly NOT built here,
per the roadmap's own item breakdown).

WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT
-----------------------------------------------------
A single interface — `LLMProvider.complete(messages, model=None)` — that
three concrete backends (Ollama, an OpenAI-compatible endpoint, an
Anthropic-compatible endpoint) each implement, so a future gateway endpoint
can call any of them without knowing which one it's talking to. It is NOT
that gateway endpoint: nothing here is wired into api/main.py, no
`/api/v1/*` route exists yet, and no audit event is written for a proxied
call. That is deliberate — "provider abstraction" and "request forwarding"
are two separate roadmap line items for a reason: this piece is
independently testable (every provider here is tested against mocked HTTP
responses, no live network needed) and independently useful even before a
gateway endpoint exists to call it.

WHY RAW `requests`, NOT THE VENDOR SDKS
------------------------------------------
Matches this project's existing convention (core/semantic_judge.py talks to
Ollama via raw `requests`, not an SDK) and avoids three new dependencies
for what is, in each case, one HTTP POST with a JSON body. "OpenAI-
compatible" is also a real, load-bearing distinction — it means anything
that speaks the same wire format (real OpenAI, Groq, Together, a local
vLLM/LM Studio server), which is exactly what pointing `OPENAI_BASE_URL`
at a different host already gives you, and an SDK's job of abstracting the
JSON shape from you is the one thing this module wants to see anyway (to
normalise it into `LLMResponse`).

ERROR MODEL, DELIBERATELY SIMPLE
-----------------------------------
One exception type, `LLMProviderError`, for every failure mode (network
error, timeout, non-2xx response, unparseable body). No retry, no circuit
breaker, no per-provider exception hierarchy — those are exactly what the
roadmap's separate "Timeout/failure handling, fallback" item is for, at
the GATEWAY level, once there is a gateway to apply them across multiple
provider attempts. A provider class raising immediately and clearly on any
failure is the correct building block for that; building partial
resilience into the block itself would just be resilience logic with no
caller yet to exercise it correctly.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30


class LLMProviderError(Exception):
    """Raised for any provider failure: network error, timeout, non-2xx
    response, or a response body that doesn't match what the provider is
    documented to return. Callers should treat this as "the call did not
    happen", the same fail-closed posture core/semantic_judge.py already
    applies to judge calls — never fabricate a completion on failure."""


@dataclass
class LLMResponse:
    """
    Normalised across all three providers, so a caller never branches on
    which one it talked to.

    `usage` is passed through verbatim from whichever provider returned it
    (each has its own field names — see each provider's docstring) and is
    NOT yet consumed by anything: Phase 5's "token accounting" roadmap item
    is what would build a policy on top of this field. Present here only
    because it costs nothing to surface what the provider already sent.

    `raw` is the full, unmodified provider response — kept for a future
    audit trail (another separate roadmap item) to log verbatim, not
    inspected by anything in this module itself.
    """
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


class LLMProvider:
    """Base interface. Not abstract via `abc` deliberately — this project's
    other pluggable-backend classes (core/detectors.py's `Detector`) use
    plain-method-not-implemented rather than ABCMeta, and consistency with
    an established pattern beats a marginally stricter enforcement here."""

    name = "base"

    def complete(self, messages: List[Dict[str, str]], model: str = None,
                timeout: float = DEFAULT_TIMEOUT_SECONDS) -> LLMResponse:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """
    Talks to Ollama's native `/api/chat` endpoint — the same endpoint
    `core/semantic_judge.py::_judge_via_llama_guard` already uses for judge
    calls, but this class is a general-purpose chat completion, not a
    safety classifier: no system-instruction injection, no verdict
    parsing, just messages in and content out.
    """
    name = "ollama"

    def __init__(self, base_url: str = None, default_model: str = None):
        self.base_url = base_url or settings.OLLAMA_CHAT_URL
        self.default_model = default_model or settings.OLLAMA_MODEL

    def complete(self, messages, model=None, timeout=DEFAULT_TIMEOUT_SECONDS) -> LLMResponse:
        model = model or self.default_model
        payload = {"model": model, "messages": messages, "stream": False}

        try:
            response = requests.post(self.base_url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            raise LLMProviderError(f"Ollama request failed: {type(e).__name__}: {e}") from e

        if response.status_code != 200:
            raise LLMProviderError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError) as e:
            raise LLMProviderError(f"Unrecognised Ollama response shape: {e}") from e

        return LLMResponse(
            content=content, model=model, provider=self.name,
            usage=None,  # Ollama's /api/chat does not report token usage.
            raw=data,
        )


class OpenAICompatibleProvider(LLMProvider):
    """
    Talks to any endpoint implementing OpenAI's `/chat/completions` wire
    format — real OpenAI at the default base URL, or any compatible
    backend (Groq, Together, a local vLLM/LM Studio server) by pointing
    `base_url` elsewhere. This is exactly what makes "OpenAI-compatible" a
    real category rather than "OpenAI specifically".
    """
    name = "openai_compatible"

    def __init__(self, api_key: str = None, base_url: str = None, default_model: str = None):
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.OPENAI_MODEL

    def complete(self, messages, model=None, timeout=DEFAULT_TIMEOUT_SECONDS) -> LLMResponse:
        model = model or self.default_model
        if not model:
            raise LLMProviderError("No model specified and no default configured (OPENAI_MODEL).")
        if not self.api_key:
            raise LLMProviderError("No API key configured (OPENAI_API_KEY).")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages}

        try:
            response = requests.post(f"{self.base_url}/chat/completions",
                                     json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise LLMProviderError(f"OpenAI-compatible request failed: {type(e).__name__}: {e}") from e

        if response.status_code != 200:
            raise LLMProviderError(f"OpenAI-compatible endpoint returned HTTP "
                                   f"{response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as e:
            raise LLMProviderError(f"Unrecognised OpenAI-compatible response shape: {e}") from e

        return LLMResponse(
            content=content, model=data.get("model", model), provider=self.name,
            usage=data.get("usage"), raw=data,
        )


class AnthropicCompatibleProvider(LLMProvider):
    """
    Talks to Anthropic's `/messages` endpoint. "Compatible" the same way
    the OpenAI provider is: `base_url` can point anywhere speaking the same
    wire format, though in practice Anthropic's Messages API is less
    commonly reimplemented by third parties than OpenAI's.
    """
    name = "anthropic_compatible"

    def __init__(self, api_key: str = None, base_url: str = None, default_model: str = None,
                anthropic_version: str = None):
        self.api_key = api_key if api_key is not None else settings.ANTHROPIC_API_KEY
        self.base_url = (base_url or settings.ANTHROPIC_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.ANTHROPIC_MODEL
        self.anthropic_version = anthropic_version or settings.ANTHROPIC_VERSION

    def complete(self, messages, model=None, timeout=DEFAULT_TIMEOUT_SECONDS) -> LLMResponse:
        model = model or self.default_model
        if not model:
            raise LLMProviderError("No model specified and no default configured (ANTHROPIC_MODEL).")
        if not self.api_key:
            raise LLMProviderError("No API key configured (ANTHROPIC_API_KEY).")

        # Anthropic's Messages API takes `system` as a top-level field, not
        # a message with role="system" — the one real shape difference from
        # the other two providers this abstraction has to paper over.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat_messages = [m for m in messages if m.get("role") != "system"]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": chat_messages, "max_tokens": 4096}
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        try:
            response = requests.post(f"{self.base_url}/messages",
                                     json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise LLMProviderError(f"Anthropic-compatible request failed: {type(e).__name__}: {e}") from e

        if response.status_code != 200:
            raise LLMProviderError(f"Anthropic-compatible endpoint returned HTTP "
                                   f"{response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            content = data["content"][0]["text"]
        except (ValueError, KeyError, IndexError) as e:
            raise LLMProviderError(f"Unrecognised Anthropic-compatible response shape: {e}") from e

        return LLMResponse(
            content=content, model=data.get("model", model), provider=self.name,
            usage=data.get("usage"), raw=data,
        )


_PROVIDER_CLASSES = {
    "ollama": OllamaProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic_compatible": AnthropicCompatibleProvider,
}


def get_provider(name: str) -> LLMProvider:
    """
    Constructs a provider by name, configured from `settings` — mirrors
    `core.detectors.get_detector`'s registry-lookup shape for consistency
    with this codebase's other pluggable-backend module.
    """
    if name not in _PROVIDER_CLASSES:
        raise KeyError(f"unknown provider '{name}'; available: {sorted(_PROVIDER_CLASSES)}")
    return _PROVIDER_CLASSES[name]()


def list_provider_names() -> list[str]:
    """The real, currently-supported provider type names -- for the
    Developer UI's model gateway view, so it lists what this deployment
    can actually route to rather than a hand-maintained, driftable copy
    of `_PROVIDER_CLASSES`' keys."""
    return sorted(_PROVIDER_CLASSES)
