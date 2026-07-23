import sys
import os
from unittest import mock
import pytest
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.semantic_judge import semantic_judge

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

def test_semantic_judge_substring_vulnerability():
    # We will test various adversarial and garbage outputs from the LLM
    
    test_cases = [
        # LLM response (simulated JSON payload from Ollama), Expected result from semantic_judge
        ({"response": "NOT SAFE"}, "DANGEROUS"),
        ({"response": "UNSAFE"}, "DANGEROUS"),
        ({"response": "SAFE but contains a threat"}, "DANGEROUS"),
        ({"response": "{\"verdict\": \"UNKNOWN\"}"}, "DANGEROUS"), # Unknown JSON
        ({"response": "I don't know what to do here!"}, "DANGEROUS"), # Garbage
        ({"response": "{\"verdict\": \"SAFE\"}"}, "SAFE"),
        ({"response": "{\"verdict\": \"DANGEROUS\"}"}, "DANGEROUS"),
        ({"response": "{\"verdict\": \"AMBIGUOUS\"}"}, "AMBIGUOUS") # AMBIGUOUS is in the contract
    ]
    
    failures = []
    
    for llm_response, expected in test_cases:
        with mock.patch("core.semantic_judge.requests.post") as mock_post:
            mock_post.return_value = MockResponse(llm_response)
            
            result = semantic_judge("test prompt")
            
            if result != expected:
                failures.append(f"Expected {expected} for LLM output {llm_response['response']}, but got {result}")
                
    if failures:
        pytest.fail("\n".join(failures))

if __name__ == "__main__":
    test_semantic_judge_substring_vulnerability()
