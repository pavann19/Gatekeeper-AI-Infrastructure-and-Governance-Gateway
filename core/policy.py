# core/policy.py
import json
import os
from core.config import POLICY_RULES_FILE

# Global cache for policies
POLICY_RULES = None

def load_policies():
    global POLICY_RULES
    try:
        # Assuming policy_rules.json is in the project root (parent of core/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, POLICY_RULES_FILE)
        
        with open(file_path, 'r') as f:
            POLICY_RULES = json.load(f)
            print("✅ Policy Rules Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Policy Load Error: {e}")
        POLICY_RULES = None

# Load on module import
load_policies()

def policy_decision(role: str, risk: str):
    """
    Determines action based on role and risk using loaded policies.
    Fallback: BLOCK if policies missing or role/risk undefined.
    """
    # Fail-safe default
    if not POLICY_RULES or "policies" not in POLICY_RULES:
        return "BLOCK", "System Error: Policies not loaded"

    user_policy = POLICY_RULES["policies"].get(role)
    
    if not user_policy:
        return POLICY_RULES.get("default_action", "BLOCK"), f"Role '{role}' not defined"
        
    action = user_policy.get(risk, POLICY_RULES.get("default_action", "BLOCK"))
    
    return action, f"Policy applied for {role} (Risk: {risk})"
