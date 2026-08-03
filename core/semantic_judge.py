import requests
import json
from core.config import OLLAMA_API_URL, OLLAMA_MODEL
from core.circuit_breaker import ollama_judge_breaker


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


def semantic_judge(prompt: str) -> str:
    """
    Asks a local LLM (Mistral) to classify the prompt as SAFE or DANGEROUS.
    Used as a tie-breaker for 'Medium Risk' prompts.

    Guarded by a circuit breaker (core/circuit_breaker.py): after repeated
    failures this returns JUDGE_OFFLINE immediately, without attempting the
    network call, rather than making every subsequent ambiguous-zone request
    wait out the same 30s timeout individually while Ollama is down.
    """
    if ollama_judge_breaker.is_open():
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
    Asks a local LLM (Mistral) to classify the LLM's own generated response 
    for toxicity, harmful content, or hallucinated dangerous instructions.
    """
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