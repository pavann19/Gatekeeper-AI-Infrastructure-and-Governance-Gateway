"""
Diagnostics and smoke-testing tool for live outbound LLM providers (Phase 5).
Tests configured external endpoints (Ollama, OpenAI-compatible, Anthropic-compatible)
with live or sandbox API credentials, reporting latency, response preview, and token accounting.

Usage:
    python -m scripts.exercise_live_providers --provider openai_compatible --prompt "Hello Gatekeeper"
    python -m scripts.exercise_live_providers --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

from core.config import settings
from core.llm_providers import (
    LLMProviderError,
    get_provider,
    list_provider_names,
)
from core.token_quota import extract_total_tokens


def exercise_provider(provider_name: str, prompt: str = "Say hello in one sentence.", model: str = None, timeout: float = 15.0) -> Dict[str, Any]:
    """Exercises a single provider, returning timing, token usage, and status."""
    try:
        provider = get_provider(provider_name)
    except Exception as e:
        return {
            "provider": provider_name,
            "status": "error",
            "error": f"Failed to instantiate provider: {e}",
            "latency_ms": 0.0,
        }

    # Redacted info
    base_url = getattr(provider, "base_url", "default")
    api_key_set = bool(getattr(provider, "api_key", None))
    target_model = model or getattr(provider, "default_model", None) or "default"

    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    try:
        resp = provider.complete(messages=messages, model=model, timeout=timeout)
        duration_ms = (time.perf_counter() - start) * 1000.0
        tokens = extract_total_tokens(resp.usage)
        return {
            "provider": provider_name,
            "status": "success",
            "model": resp.model,
            "base_url": base_url,
            "api_key_configured": api_key_set,
            "latency_ms": round(duration_ms, 2),
            "tokens_used": tokens,
            "usage_raw": resp.usage,
            "content_preview": resp.content[:150] if resp.content else "",
        }
    except LLMProviderError as e:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "provider": provider_name,
            "status": "failed",
            "error": str(e),
            "base_url": base_url,
            "api_key_configured": api_key_set,
            "latency_ms": round(duration_ms, 2),
        }
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "provider": provider_name,
            "status": "unhandled_error",
            "error": f"{type(e).__name__}: {e}",
            "latency_ms": round(duration_ms, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Live LLM Provider Smoke Test Harness")
    parser.add_argument("--provider", choices=list_provider_names() + ["all"], default="all",
                        help="Provider to exercise (default: all)")
    parser.add_argument("--prompt", default="State in 5 words that you are online.",
                        help="Prompt to send")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    args = parser.parse_args()

    targets = list_provider_names() if args.provider == "all" else [args.provider]
    results = []

    for name in targets:
        res = exercise_provider(name, prompt=args.prompt, model=args.model, timeout=args.timeout)
        results.append(res)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 60)
    print("GATEKEEPER LIVE LLM PROVIDER SMOKE TESTS")
    print("=" * 60)
    for r in results:
        status_sym = "[OK]" if r["status"] == "success" else "[SKIP/FAIL]"
        print(f"\nProvider: {r['provider']} {status_sym}")
        print(f"  Status: {r['status']}")
        print(f"  Latency: {r['latency_ms']} ms")
        if r["status"] == "success":
            print(f"  Model: {r.get('model')}")
            print(f"  Tokens: {r.get('tokens_used')}")
            print(f"  Content: {r.get('content_preview')}")
        else:
            print(f"  Error: {r.get('error')}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
