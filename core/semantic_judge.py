import requests
import json

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
            "model": "mistral",
            "prompt": f"{system_instruction}\n\nUSER PROMPT: {prompt}",
            "stream": False
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)

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