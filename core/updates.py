# core/updates.py
import requests
import json
import os
import time
from core.embeddings import get_embedding, cosine_similarity
from core.logger import get_logger

logger = get_logger(__name__)

# --- THREAT INTELLIGENCE CONFIGURATION ---
# NeMo Alignment: Updating Rails
LOCAL_THREAT_DB = "threat_feed.json"
THREAT_SOURCES = [
    # Community maintained Jailbreak list (DAN)
    "https://raw.githubusercontent.com/0xk1h0/ChatGPT_DAN/main/README.md",
]

# In-memory storage
DYNAMIC_THREATS = []
DYNAMIC_SAFE_HARBORS = []

def load_local_threats():
    """Loads persisted threat signatures from disk."""
    if os.path.exists(LOCAL_THREAT_DB):
        try:
            with open(LOCAL_THREAT_DB, 'r') as f:
                data = json.load(f)
                # Handle potential format changes if needed
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to load local threat DB: {e}")
            return []
    return []

def save_local_threats(threats):
    """Persists new threats to disk to survive restarts."""
    try:
        with open(LOCAL_THREAT_DB, 'w') as f:
            json.dump(threats, f)
    except Exception as e:
        logger.error(f"Failed to save threat DB: {e}")

# Initialize memory with disk content on module load
DYNAMIC_THREATS = load_local_threats()

def fetch_latest_threats():
    """
    Connects to GitHub to fetch the latest Jailbreak signatures.
    Returns: (count of new threats, success status)
    """
    global DYNAMIC_THREATS
    new_count = 0
    
    logger.info("🌍 CONNECTING to Open Threat Intelligence Feeds...")
    
    for url in THREAT_SOURCES:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                raw_text = response.text
                
                # Heuristic Parsing: Grab paragraphs that look like prompts (long text)
                # In a real system, we would use specific parsers for STIX/TAXII or JSON feeds.
                lines = [line.strip() for line in raw_text.split('\n') if len(line) > 60]
                
                # Take top candidates to avoid bloat in this research demo
                candidates = lines[:5]
                
                for text in candidates:
                    # Deduplication: Check if text already exists
                    if not any(t.get('text') == text for t in DYNAMIC_THREATS):
                        # Convert tensor to list for JSON serialization
                        vec = get_embedding(text).tolist() 
                        
                        entry = {
                            "text": text, 
                            "vector": vec, 
                            "source": url, 
                            "date": time.time()
                        }
                        DYNAMIC_THREATS.append(entry)
                        new_count += 1
                        
            else:
                logger.warning(f"Feed Unreachable ({response.status_code}): {url}")
                
        except Exception as e:
            logger.error(f"Connection Failed ({url}): {e}")

    if new_count > 0:
        save_local_threats(DYNAMIC_THREATS)
        logger.info(f"THREAT INTEL UPDATED: Added {new_count} new signatures.")
        return new_count, True
    else:
        logger.info("Threat Feed is up to date.")
        return 0, True

def check_dynamic_threats(prompt_vec):
    """
    Checks if the prompt matches any newly learned threats from GitHub.
    Returns the highest similarity score found.
    """
    max_score = 0.0
    # Ensure DYNAMIC_THREATS is loaded
    if not DYNAMIC_THREATS:
        return 0.0
        
    for threat in DYNAMIC_THREATS:
        # threat['vector'] is a list from JSON, cosine_similarity handles it
        score = cosine_similarity(prompt_vec, threat["vector"])
        if score > max_score:
            max_score = score
            
    return max_score

def check_dynamic_safe_harbors(prompt_vec):
    """
    Checks if the prompt matches any newly learned safe harbors.
    (Currently empty, but structured for future expansion).
    """
    max_score = 0.0
    for safe in DYNAMIC_SAFE_HARBORS:
        score = cosine_similarity(prompt_vec, safe["vector"])
        if score > max_score:
            max_score = score
            
    return max_score