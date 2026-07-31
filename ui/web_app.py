import streamlit as st
import time
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gatekeeper - AI Governance",
    page_icon="🛡️",
    layout="centered"
)

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.success("✅ System Online")
    st.info("🔒 Privacy Shield: Active (GDPR)")
    st.warning("🛡️ Defense Level: Neuro-Symbolic")
    
    st.divider()
    
    # Capability is resolved SERVER-SIDE from the API key, never chosen by the
    # client. The old "Simulate User Role" dropdown sent the tier in the request
    # body, which the gateway then trusted -- a privilege escalation, since
    # INTERNAL maps HIGH -> ALLOW. Supplying a key is now the only way to raise
    # privilege, and an empty box means anonymous (GENERAL).
    api_key = st.text_input(
        "API Key (optional)",
        type="password",
        help="Presented as 'Authorization: Bearer <key>'. Leave blank for "
             "anonymous access at GENERAL. Issue keys with: "
             "python -m scripts.manage_api_keys issue",
    )
    
    st.divider()
    
    if st.button("🧹 Flush Semantic Cache"):
        try:
            resp = requests.post(f"{API_URL}/cache/flush")
            if resp.status_code == 200:
                st.toast("Memory cleared!", icon="🧹")
            else:
                st.error("Failed to flush cache.")
        except Exception as e:
            st.error(f"API Connection Error: {e}")

    st.header("📡 Live Intelligence")
    if st.button("🌍 Fetch GitHub Threat Feed"):
        with st.spinner("Downloading Jailbreak Signatures from GitHub..."):
            try:
                resp = requests.post(f"{API_URL}/update")
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ Synced! Learned {data.get('signatures_added', 0)} attack patterns.", icon="🛡️")
                else:
                    st.error("❌ Connection failed.", icon="⚠️")
            except Exception as e:
                st.error(f"API Connection Error: {e}")

# --- MAIN INTERFACE ---
st.title("🛡️ Gatekeeper")
st.markdown("**Enterprise AI Governance Gateway** | *Compliant with IBM 2025 Security Framework*")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT HANDLING ---
user_query = st.chat_input("Enter your prompt here...")

if user_query:
    # 0. Show User Prompt
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 1. & 2. & 3. PRIVACY + SECURITY + POLICY VIA API GATEWAY
    with st.spinner("🛡️ Gatekeeper Gateway: Analyzing Intent & Security Risks..."):
        try:
            api_response = requests.post(
                f"{API_URL}/assess",
                json={"prompt": user_query},
                headers=({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            )
            api_response.raise_for_status()
            result = api_response.json()
            
            risk_level = result["risk_level"]
            decision = result["decision"]
            details = result["details"]
            clean_query = result["clean_prompt"]
            redacted_items = result.get("redacted_items", [])
            
        except requests.exceptions.RequestException as e:
            st.error(f"🔌 **API Connection Error:** Gatekeeper Gateway is offline. Ensure the FastAPI server is running. Error: {e}")
            st.stop()
            
    # Show Privacy actions if any
    if redacted_items:
        with st.expander("🔒 Privacy Action Triggered", expanded=False):
            st.info(f"Redacted {len(redacted_items)} sensitive items.")
            st.code(f"Cleaned: {clean_query}")

    # --- EXECUTION LAYER ---
    if decision == "BLOCK":
        msg = f"❌ **BLOCKED:** Security Policy Violation ({details.get('source', 'Unknown')})\n\n*Reason: {details.get('policy_reason', '')}*"
        st.error(msg)
        with st.expander("View Forensic Logs"):
            st.json(details)
        st.session_state.messages.append({"role": "assistant", "content": msg})

    elif decision == "RESTRICT":
        msg = f"⚠️ **RESTRICTED:** {details.get('policy_reason', 'Topic Restricted')}\n\n*I cannot provide actionable instructions, but I can discuss general concepts.*"
        st.warning(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})

    else:
        # ✅ ALLOWED -> Call Mistral
        st.success(f"✅ **ALLOWED:** Forwarding to Model (Risk Score: {details.get('semantic_score', 0.0):.3f})")
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Payload for Ollama
                payload = {"model": OLLAMA_MODEL, "prompt": clean_query, "stream": True}
                
                # HTTP Request to Local LLM
                with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line)
                                token = data.get("response", "")
                                full_response += token
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"⚠️ Model Error: {response.status_code}")
            except Exception as e:
                st.error(f"🔌 **Connection Error:** Ollama is offline. Run 'ollama serve'. Error: {e}")
