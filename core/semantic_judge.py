import requests
import json
from core.config import OLLAMA_API_URL, OLLAMA_MODEL


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
        base_url = "/".join(OLLAMA_API_URL.split("/")[:-2])
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
    """
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
                    return verdict
                else:
                    # Fail closed on unknown string
                    return "DANGEROUS"
            except json.JSONDecodeError:
                # Fail closed on JSON parse error
                return "DANGEROUS"

        # LLM returned non-200 -> fail closed.
        return "DANGEROUS"

    except Exception:
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