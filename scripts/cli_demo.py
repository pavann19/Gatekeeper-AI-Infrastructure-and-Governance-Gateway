# app.py
from core.risk import assess_risk
from core.policy import policy_decision
from core.logger import log_event
from core.llm import generate_llm_response_stream
from core.auth import get_user_role
from core.multimodal import normalize_image, normalize_audio
from core.updates import fetch_latest_threats

def generate_safe_response(prompt):
    return "Here is general, high-level information without actionable steps. (Educational Mode)"

def handle_prompt(prompt, role):
    # 1. Assess Risk
    risk, metadata = assess_risk(prompt)
    
    # 2. Make Policy Decision
    decision, reason = policy_decision(role, risk)

    # 3. Log with Metadata
    log_event(role, prompt, risk, decision, metadata)

    if decision == "BLOCK":
        return f"❌ BLOCKED\nReason: {reason}\nSource: {metadata.get('source')}"

    if decision == "RESTRICT":
        return f"⚠️ RESTRICTED\nReason: {reason}\nSource: {metadata.get('source')}\n\n{generate_safe_response(prompt)}"

    print("🧠 LLM Response:\n")
    generate_llm_response_stream(prompt)
    return ""

def handle_multimodal_request(input_str, role):
    """
    Simulates Multi-Modal Input.
    Usage: Type 'image:path/to/img' or 'audio:path/to/wav' in the prompt.
    """
    normalized_prompt = input_str
    
    if input_str.startswith("image:"):
        file_path = input_str.split(":", 1)[1]
        normalized_prompt = normalize_image(file_path)
        print(f"📷 Image Input Normalized to: '{normalized_prompt}'")
        
    elif input_str.startswith("audio:"):
        file_path = input_str.split(":", 1)[1]
        normalized_prompt = normalize_audio(file_path)
        print(f"🎤 Audio Input Normalized to: '{normalized_prompt}'")

    return handle_prompt(normalized_prompt, role)

if __name__ == "__main__":
    print("\n🛡️  Gatekeeper AI Governance Gateway | v2.0 NeMo-Aligned")
    print("--------------------------------------------------")
    print("COMMANDS:")
    print(" - Type text for standard prompt")
    print(" - 'image:<path>' for Image Rail test")
    print(" - 'audio:<path>' for Audio Rail test")
    print(" - 'update' to fetch Threat Feeds")
    print("--------------------------------------------------\n")

    role = get_user_role()
    
    while True:
        try:
            user_input = input(f"[{role}] >> ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if user_input.lower() == "update":
                fetch_latest_threats()
                continue
                
            print(handle_multimodal_request(user_input, role))
            print("-" * 30)
            
        except KeyboardInterrupt:
            break