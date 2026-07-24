import sys
import os
from unittest import mock
import pytest
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.semantic_judge import judge_available, semantic_judge

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


def test_judge_availability_probes_the_api_tags_endpoint():
    """
    Regression: judge_available built the probe URL by stripping TWO path
    segments off OLLAMA_API_URL (.../api/generate -> .../ ) and appending
    '/tags', producing '.../tags' — a 404 against any real Ollama server, which
    made the methodology gate report the judge offline even when it was up. It
    must hit '.../api/tags'.
    """
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return MockResponse({"models": [{"name": "llama3.2:latest"}]})

    with mock.patch("core.semantic_judge.requests.get", side_effect=fake_get):
        with mock.patch("core.semantic_judge.OLLAMA_API_URL",
                        "http://localhost:11434/api/generate"):
            with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
                ok, detail = judge_available()

    assert captured["url"] == "http://localhost:11434/api/tags", (
        f"probe hit {captured['url']!r}, not the /api/tags endpoint"
    )
    assert ok is True, detail


def test_judge_unavailable_when_model_absent():
    """A reachable server missing the configured model is not 'available'."""
    def fake_get(url, timeout=None):
        return MockResponse({"models": [{"name": "some-other-model:latest"}]})

    with mock.patch("core.semantic_judge.requests.get", side_effect=fake_get):
        with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
            ok, detail = judge_available()

    assert ok is False
    assert "not present" in detail

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
