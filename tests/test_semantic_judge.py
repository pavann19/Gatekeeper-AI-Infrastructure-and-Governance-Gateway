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
    # We will test various adversarial and garbage outputs from the LLM.
    # Pinned to a non-Llama-Guard model: semantic_judge() branches on
    # uses_llama_guard_protocol(OLLAMA_MODEL) and takes a completely
    # different parsing path (_judge_via_llama_guard, its own native
    # safe/unsafe format) for Llama Guard models -- this test targets the
    # GENERIC parsing path specifically, so it must not depend on whichever
    # model happens to be the ambient default (see
    # docs/ENGINEERING_ASSESSMENT.md's OLLAMA_MODEL default fix, 2026-08-16).

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

    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
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


# --- Llama Guard judge protocol ---------------------------------------------
#
# Llama Guard is a fine-tuned classifier, not an instructable chat model: it
# emits `safe` / `unsafe\nS<n>` and ignores output-format instructions
# entirely. Verified against the real model. Routing it through the general
# JSON path makes json.loads() raise on EVERY prompt, failing closed to
# DANGEROUS every time — a judge that blocks everything. These tests pin the
# separate protocol path that prevents that.

from core.semantic_judge import _judge_via_llama_guard, uses_llama_guard_protocol


@pytest.mark.parametrize("name,expected", [
    ("llama-guard3", True),
    ("llama-guard3:8b", True),
    ("meta-llama/Llama-Guard-3-8B", True),
    ("LLAMA-GUARD3", True),
    ("llama3.2", False),
    ("mistral", False),
    ("gpt-oss", False),
])
def test_llama_guard_protocol_detection(name, expected):
    assert uses_llama_guard_protocol(name) is expected


def _guard_response(content, status=200):
    return MockResponse({"message": {"content": content}}, status_code=status)


def test_llama_guard_safe_verdict():
    with mock.patch("core.semantic_judge.requests.post",
                    return_value=_guard_response("safe")):
        assert _judge_via_llama_guard("harmless") == "SAFE"


def test_llama_guard_unsafe_verdict_with_category():
    with mock.patch("core.semantic_judge.requests.post",
                    return_value=_guard_response("unsafe\nS9")):
        assert _judge_via_llama_guard("bomb instructions") == "DANGEROUS"


def test_llama_guard_uses_chat_endpoint_with_bare_user_prompt():
    """
    Two correctness requirements in one: the chat endpoint (so the model's
    own template positions the content correctly) and NO system instruction
    prepended. The general path's instruction text contains 'violence,
    illegal acts, hacking' — sent to Llama Guard it would classify OUR
    instruction alongside the user's prompt.
    """
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _guard_response("safe")

    with mock.patch("core.semantic_judge.requests.post", side_effect=fake_post):
        _judge_via_llama_guard("my actual prompt")

    assert captured["url"].endswith("/api/chat")
    messages = captured["json"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "my actual prompt"  # verbatim, unwrapped


def test_llama_guard_unrecognised_output_fails_closed():
    with mock.patch("core.semantic_judge.requests.post",
                    return_value=_guard_response("I think this might be okay?")):
        assert _judge_via_llama_guard("x") == "DANGEROUS"


def test_llama_guard_non_200_fails_closed():
    with mock.patch("core.semantic_judge.requests.post",
                    return_value=_guard_response("safe", status=500)):
        assert _judge_via_llama_guard("x") == "DANGEROUS"


def test_semantic_judge_routes_guard_models_to_the_guard_path():
    """The dispatch itself: a guard model must never reach the JSON parser."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama-guard3"):
        with mock.patch("core.semantic_judge._judge_via_llama_guard",
                        return_value="SAFE") as guard_path:
            with mock.patch("core.semantic_judge.requests.post") as json_path:
                assert semantic_judge("anything") == "SAFE"
                guard_path.assert_called_once()
                json_path.assert_not_called()


def test_semantic_judge_keeps_chat_models_on_the_json_path():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge._judge_via_llama_guard") as guard_path:
            with mock.patch("core.semantic_judge.requests.post",
                            return_value=MockResponse({"response": '{"verdict": "SAFE"}'})):
                assert semantic_judge("anything") == "SAFE"
                guard_path.assert_not_called()


def test_guard_path_exception_is_judge_offline_not_a_crash():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama-guard3"):
        with mock.patch("core.semantic_judge._judge_via_llama_guard",
                        side_effect=ConnectionError("refused")):
            assert semantic_judge("anything") == "JUDGE_OFFLINE"


# ---------------------------------------------------------------------------
# output_judge: the SAME protocol bug semantic_judge already had, found
# during Phase 2 (Output Security) after OLLAMA_MODEL's default changed to
# a Llama Guard variant. output_judge never got the equivalent fix, so it
# always sent the generic instructable-chat prompt regardless of model —
# under a Llama Guard default that means json.loads() on "safe"/"unsafe"
# raises every time and every response fails closed to DANGEROUS.
# ---------------------------------------------------------------------------

from core.semantic_judge import output_judge


def test_output_judge_routes_guard_models_to_the_guard_path():
    """The dispatch itself: a guard model must never reach the JSON parser."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama-guard3"):
        with mock.patch("core.semantic_judge._judge_via_llama_guard",
                        return_value="SAFE") as guard_path:
            with mock.patch("core.semantic_judge.requests.post") as json_path:
                assert output_judge("a perfectly ordinary response") == "SAFE"
                guard_path.assert_called_once_with("a perfectly ordinary response")
                json_path.assert_not_called()


def test_output_judge_keeps_chat_models_on_the_json_path():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge._judge_via_llama_guard") as guard_path:
            with mock.patch("core.semantic_judge.requests.post",
                            return_value=MockResponse({"response": '{"verdict": "SAFE"}'})):
                assert output_judge("anything") == "SAFE"
                guard_path.assert_not_called()


def test_output_judge_under_guard_default_does_not_fail_closed_on_every_response():
    """
    THE REGRESSION THIS GUARDS: before the fix, this exact scenario --
    Llama Guard configured, a genuinely safe response -- returned DANGEROUS
    unconditionally because the raw "safe" text isn't valid JSON. Simulates
    Llama Guard's real /api/chat response shape (see _judge_via_llama_guard),
    not a mock of output_judge's own internals, so it exercises the real
    dispatch and the real Llama Guard response parser.
    """
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama-guard3"):
        with mock.patch(
            "core.semantic_judge.requests.post",
            return_value=MockResponse({"message": {"content": "safe"}}),
        ):
            assert output_judge("Here is a recipe for banana bread.") == "SAFE"
