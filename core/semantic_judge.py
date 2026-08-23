import requests
import json
from core.config import OLLAMA_API_URL, OLLAMA_MODEL
from core.circuit_breaker import ollama_judge_breaker
from core.logger import get_logger

logger = get_logger(__name__)


def judge_available() -> tuple:
    """
    Probes the local judge backend for reachability.

    Evaluation harnesses MUST call this before measuring classifier quality.
    `judge_arbitration` fails closed to HIGH when the judge is unreachable,
    so benchmarking against an offline judge silently measures sidecar
    availability rather than detection accuracy — which is exactly how the
    pre-fix benchmark produced an uninterpretable 98% false-positive rate.

    Returns (available: bool, detail: str).
    """
    try:
        # OLLAMA_API_URL ends in ".../api/generate"; the model list lives at
        # ".../api/tags". Drop only the final segment ("generate") so the "api"
        # segment is preserved — dropping two produced ".../tags" (404).
        base_url = "/".join(OLLAMA_API_URL.split("/")[:-1])
        resp = requests.get(f"{base_url}/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"judge endpoint returned HTTP {resp.status_code}"
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(m.split(":")[0] == OLLAMA_MODEL for m in models):
            return False, (
                f"model '{OLLAMA_MODEL}' not present on the judge host "
                f"(available: {models or 'none'})"
            )
        return True, f"judge reachable, model '{OLLAMA_MODEL}' loaded"
    except Exception as e:
        return False, f"judge unreachable at {OLLAMA_API_URL}: {type(e).__name__}: {e}"


def uses_llama_guard_protocol(model_name: str) -> bool:
    """
    True when the configured judge model is a Llama Guard variant, which
    speaks a fundamentally different protocol from a general chat model.

    Detection is by name because that is what actually distinguishes them at
    configuration time — `llama-guard3`, `llama-guard3:8b`,
    `meta-llama/Llama-Guard-3-8B` all match, `llama3.2` and `mistral` do not.
    """
    return "guard" in model_name.lower()


def _judge_via_llama_guard(prompt: str) -> str:
    """
    Judges using a Llama Guard model served by Ollama.

    WHY THIS IS A SEPARATE PATH, not a prompt tweak. Llama Guard is a
    fine-tuned safety classifier, not an instructable chat model: it emits
    exactly `safe` or `unsafe\\nS<n>` and IGNORES any output-format
    instruction you give it. Verified against the real model — asking it for
    JSON returns `safe` regardless, so the general path's `json.loads()`
    raises and fails closed to DANGEROUS on EVERY prompt, turning every
    ambiguous request into HIGH. A judge that blocks everything is not a
    working judge.

    Two further differences that make this a protocol change rather than a
    parser change:

    1. It calls `/api/chat`, not `/api/generate`. Llama Guard's accuracy
       depends on its own chat template placing the user content in a
       specific slot. The general path prepends a system instruction that
       itself contains the words "violence, illegal acts, hacking" — sent
       through `/api/generate`, Llama Guard would be classifying OUR
       INSTRUCTION TEXT alongside the user's prompt, which is both wrong and
       likely to skew toward unsafe. `/api/chat` sends only the user prompt.

    2. There is no AMBIGUOUS verdict. Llama Guard is binary. Where the
       general chat judge can hedge (and `semantic_judge` maps that to
       MEDIUM), this returns only SAFE or DANGEROUS. That is a real
       behavioural difference for Stage 4, not a bug — a purpose-built
       classifier committing to an answer is the point of using one.
    """
    # OLLAMA_API_URL ends in ".../api/generate"; the chat endpoint is
    # ".../api/chat". Same derivation the availability probe uses.
    base_url = "/".join(OLLAMA_API_URL.split("/")[:-1])
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(f"{base_url}/chat", json=payload, timeout=60)

    if response.status_code != 200:
        ollama_judge_breaker.record_failure()
        return "DANGEROUS"

    raw = response.json().get("message", {}).get("content", "").strip()
    verdict_line = raw.split("\n")[0].strip().lower()

    if verdict_line == "safe":
        ollama_judge_breaker.record_success()
        return "SAFE"
    if verdict_line == "unsafe":
        ollama_judge_breaker.record_success()
        # Hazard categories (S1..S13) are on the following line. Logged rather
        # than returned because Stage 4's contract is a risk level, not a
        # taxonomy — but the category is the single most useful thing in an
        # audit record for explaining WHY something was blocked.
        categories = raw.split("\n", 1)[1].strip() if "\n" in raw else ""
        logger.info(f"Llama Guard verdict: unsafe {categories}".strip())
        return "DANGEROUS"

    # Backend answered, but not in a shape Llama Guard should ever produce.
    # Fail closed, and do NOT count it against the breaker — the service is
    # up, so tripping the breaker would suppress a working judge.
    logger.error(f"Unrecognised Llama Guard output: {raw[:120]!r} — failing closed.")
    return "DANGEROUS"


def semantic_judge(prompt: str) -> str:
    """
    Asks a local LLM to classify the prompt as SAFE or DANGEROUS.
    Used as a tie-breaker for 'Medium Risk' prompts.

    Dispatches on the configured model: a Llama Guard variant speaks its own
    protocol (see `_judge_via_llama_guard`), anything else is treated as an
    instructable chat model and asked for JSON.

    Guarded by a circuit breaker (core/circuit_breaker.py): after repeated
    failures this returns JUDGE_OFFLINE immediately, without attempting the
    network call, rather than making every subsequent ambiguous-zone request
    wait out the same 30s timeout individually while Ollama is down.
    """
    if ollama_judge_breaker.is_open():
        return "JUDGE_OFFLINE"

    if uses_llama_guard_protocol(OLLAMA_MODEL):
        try:
            return _judge_via_llama_guard(prompt)
        except Exception:
            ollama_judge_breaker.record_failure()
            return "JUDGE_OFFLINE"

    system_instruction = (
        "You are an AI Safety Filter. "
        "Analyze the following user prompt for harm (violence, illegal acts, hacking). "
        "Output ONLY a valid JSON object with a single key 'verdict'. "
        "The value must be exactly one of: 'SAFE', 'DANGEROUS', or 'AMBIGUOUS'. "
        "Do not include markdown formatting or any other text."
    )

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_instruction}\n\nUSER PROMPT: {prompt}",
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            raw = response.json().get("response", "").strip()

            try:
                # Remove any potential markdown blocks if the LLM leaked them
                if raw.startswith("```json"):
                    raw = raw.replace("```json", "").replace("```", "").strip()

                parsed = json.loads(raw)
                verdict = parsed.get("verdict", "").upper()

                if verdict in {"SAFE", "DANGEROUS", "AMBIGUOUS"}:
                    ollama_judge_breaker.record_success()
                    return verdict
                else:
                    # Fail closed on unknown string. Not a backend failure —
                    # Ollama answered fine, it just didn't follow the
                    # requested format — so this does not count against the
                    # breaker.
                    return "DANGEROUS"
            except json.JSONDecodeError:
                # Same reasoning: the backend responded, it just wasn't valid
                # JSON. Not a connectivity/availability problem.
                return "DANGEROUS"

        # LLM returned non-200 -> fail closed AND count as a backend failure.
        ollama_judge_breaker.record_failure()
        return "DANGEROUS"

    except Exception:
        ollama_judge_breaker.record_failure()
        return "JUDGE_OFFLINE"

def output_judge(response_text: str) -> str:
    """
    Asks a local LLM to classify an LLM-generated response for toxicity,
    harmful content, or hallucinated dangerous instructions.

    BUG FIX (Phase 2, Output Security): this previously always sent the
    generic instructable-chat prompt below, regardless of OLLAMA_MODEL. Once
    OLLAMA_MODEL's default became a Llama Guard variant (see core/config.py's
    OLLAMA_MODEL fix), every call here hit exactly the failure
    `_judge_via_llama_guard`'s own docstring already documents for the input
    path: Llama Guard ignores output-format instructions and always replies
    `safe`/`unsafe`, so `json.loads()` on its raw text raises and this
    function failed closed to DANGEROUS on every single response — a judge
    that blocks everything is not a working judge. Same dispatch as
    `semantic_judge` fixes it the same way.
    """
    if uses_llama_guard_protocol(OLLAMA_MODEL):
        return _judge_via_llama_guard(response_text)

    system_instruction = (
        "You are an AI Output Evaluator. "
        "Analyze the following generated response for harm, toxicity, hate speech, "
        "or dangerous instructions (e.g., how to build a bomb). "
        "Output ONLY a valid JSON object with a single key 'verdict'. "
        "The value must be exactly one of: 'SAFE', 'DANGEROUS'. "
        "Do not include markdown formatting or any other text."
    )

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_instruction}\n\nGENERATED RESPONSE: {response_text}",
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            try:
                if raw.startswith("```json"):
                    raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                verdict = parsed.get("verdict", "").upper()
                if verdict in {"SAFE", "DANGEROUS"}:
                    return verdict
                return "DANGEROUS"
            except json.JSONDecodeError:
                return "DANGEROUS"
        return "DANGEROUS"

    except Exception:
        return "JUDGE_OFFLINE"