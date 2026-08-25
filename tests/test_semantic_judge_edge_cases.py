"""
Edge-case coverage for core/semantic_judge.py, additional to
tests/test_semantic_judge.py (which already covers the generic JSON path's
known-bad-string verdicts, prompt-injection delimiter wrapping, and the full
Llama Guard protocol path).

This file targets:
  - malformed LLM responses the existing suite doesn't exercise: missing
    'verdict' key, non-string verdict values, JSON that parses but isn't a
    dict, and empty/non-JSON bodies
  - timeout/connection-error propagation from the underlying requests call
  - circuit-breaker integration: is_open() short-circuits before any network
    call, and which code paths do/don't call record_success/record_failure
  - the asymmetry between semantic_judge (breaker-gated) and output_judge
    (not breaker-gated) for the generic chat-model path
"""
import os
import sys
from unittest import mock

import requests

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.semantic_judge import semantic_judge, output_judge


class MockResponse:
    def __init__(self, json_data=None, status_code=200, text_body=None):
        self._json_data = json_data
        self.status_code = status_code
        self._text_body = text_body

    def json(self):
        if self._text_body is not None:
            # Simulate requests raising on genuinely non-JSON bodies.
            raise ValueError("No JSON object could be decoded")
        return self._json_data


# --- Circuit breaker: short-circuit before any network call -----------------

def test_semantic_judge_open_breaker_skips_the_network_call_entirely():
    with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=True):
        with mock.patch("core.semantic_judge.requests.post") as mock_post:
            result = semantic_judge("anything")

    assert result == "JUDGE_OFFLINE"
    mock_post.assert_not_called()


def test_semantic_judge_closed_breaker_proceeds_to_call_the_network():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch(
                "core.semantic_judge.requests.post",
                return_value=MockResponse({"response": '{"verdict": "SAFE"}'}),
            ) as mock_post:
                result = semantic_judge("anything")

    assert result == "SAFE"
    mock_post.assert_called_once()


def test_semantic_judge_success_records_success_on_breaker():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_success") as rec_ok:
                with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                    with mock.patch(
                        "core.semantic_judge.requests.post",
                        return_value=MockResponse({"response": '{"verdict": "SAFE"}'}),
                    ):
                        result = semantic_judge("anything")

    assert result == "SAFE"
    rec_ok.assert_called_once()
    rec_fail.assert_not_called()


# --- Timeout / connection-error propagation ---------------------------------

def test_semantic_judge_timeout_fails_closed_to_judge_offline_and_records_failure():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                with mock.patch(
                    "core.semantic_judge.requests.post",
                    side_effect=requests.exceptions.Timeout("read timed out"),
                ):
                    result = semantic_judge("anything")

    assert result == "JUDGE_OFFLINE"
    rec_fail.assert_called_once()


def test_semantic_judge_connection_error_fails_closed_to_judge_offline():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch(
                "core.semantic_judge.requests.post",
                side_effect=requests.exceptions.ConnectionError("refused"),
            ):
                result = semantic_judge("anything")

    assert result == "JUDGE_OFFLINE"


def test_output_judge_timeout_fails_closed_to_judge_offline():
    """output_judge has its own try/except Exception -> JUDGE_OFFLINE, entirely
    separate from semantic_judge's; must be exercised independently."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch(
            "core.semantic_judge.requests.post",
            side_effect=requests.exceptions.Timeout("read timed out"),
        ):
            result = output_judge("some generated text")

    assert result == "JUDGE_OFFLINE"


# --- Malformed LLM response bodies ------------------------------------------

def test_semantic_judge_missing_verdict_key_fails_closed_without_breaker_penalty():
    """parsed.get('verdict', '') yields '' when the key is absent, which is
    not in the known verdict set -> DANGEROUS. This is a format mismatch, not
    a backend outage, so it must NOT count against the circuit breaker."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                with mock.patch(
                    "core.semantic_judge.requests.post",
                    return_value=MockResponse({"response": '{"other_key": "SAFE"}'}),
                ):
                    result = semantic_judge("anything")

    assert result == "DANGEROUS"
    rec_fail.assert_not_called()


def test_semantic_judge_non_string_verdict_value_fails_closed_to_judge_offline():
    """A verdict that parses as JSON but isn't a string (e.g. a number) blows
    up on .upper() with an AttributeError; that's not caught by the
    JSONDecodeError-only inner except, so it propagates to the outer
    except Exception -> JUDGE_OFFLINE, and DOES count against the breaker
    since the outer handler treats it as a genuine failure."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                with mock.patch(
                    "core.semantic_judge.requests.post",
                    return_value=MockResponse({"response": '{"verdict": 42}'}),
                ):
                    result = semantic_judge("anything")

    assert result == "JUDGE_OFFLINE"
    rec_fail.assert_called_once()


def test_semantic_judge_json_top_level_list_fails_closed_to_judge_offline():
    """Valid JSON, but not an object -> .get() doesn't exist on a list ->
    AttributeError -> outer except -> JUDGE_OFFLINE."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch(
                "core.semantic_judge.requests.post",
                return_value=MockResponse({"response": '["SAFE", "DANGEROUS"]'}),
            ):
                result = semantic_judge("anything")

    assert result == "JUDGE_OFFLINE"


def test_semantic_judge_empty_response_body_is_not_valid_json():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                with mock.patch(
                    "core.semantic_judge.requests.post",
                    return_value=MockResponse({"response": ""}),
                ):
                    result = semantic_judge("anything")

    assert result == "DANGEROUS"
    rec_fail.assert_not_called()


def test_semantic_judge_lowercase_verdict_is_normalised_and_accepted():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch(
                "core.semantic_judge.requests.post",
                return_value=MockResponse({"response": '{"verdict": "safe"}'}),
            ):
                result = semantic_judge("anything")

    assert result == "SAFE"


def test_output_judge_missing_verdict_key_fails_closed_to_dangerous():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch(
            "core.semantic_judge.requests.post",
            return_value=MockResponse({"response": '{"unexpected": "field"}'}),
        ):
            result = output_judge("some text")

    assert result == "DANGEROUS"


def test_output_judge_ambiguous_is_not_in_its_verdict_set_and_fails_closed():
    """output_judge's contract only accepts SAFE/DANGEROUS (no AMBIGUOUS,
    unlike semantic_judge) -- an AMBIGUOUS verdict here must fail closed."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch(
            "core.semantic_judge.requests.post",
            return_value=MockResponse({"response": '{"verdict": "AMBIGUOUS"}'}),
        ):
            result = output_judge("some text")

    assert result == "DANGEROUS"


# --- output_judge is not gated by the circuit breaker -----------------------

def test_output_judge_ignores_open_breaker_and_still_calls_the_network():
    """Unlike semantic_judge, output_judge never checks
    ollama_judge_breaker.is_open() -- it has no fast-fail path. Documenting
    this asymmetry so a future refactor that accidentally unifies the two
    paths shows up as a test change, not a silent behavior shift."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=True):
            with mock.patch(
                "core.semantic_judge.requests.post",
                return_value=MockResponse({"response": '{"verdict": "SAFE"}'}),
            ) as mock_post:
                result = output_judge("some text")

    assert result == "SAFE"
    mock_post.assert_called_once()


def test_output_judge_generic_path_never_touches_the_breaker():
    """The generic (non-Llama-Guard) path in output_judge doesn't call
    record_success/record_failure at all -- only semantic_judge and the
    Llama Guard path do. A non-200 response should fail closed without any
    breaker interaction."""
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.record_success") as rec_ok:
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                with mock.patch(
                    "core.semantic_judge.requests.post",
                    return_value=MockResponse({"response": '{"verdict": "SAFE"}'}, status_code=500),
                ):
                    result = output_judge("some text")

    assert result == "DANGEROUS"
    rec_ok.assert_not_called()
    rec_fail.assert_not_called()


# --- Non-200 status on the generic path -------------------------------------

def test_semantic_judge_non_200_status_records_failure_and_fails_closed():
    with mock.patch("core.semantic_judge.OLLAMA_MODEL", "llama3.2"):
        with mock.patch("core.semantic_judge.ollama_judge_breaker.is_open", return_value=False):
            with mock.patch("core.semantic_judge.ollama_judge_breaker.record_failure") as rec_fail:
                with mock.patch(
                    "core.semantic_judge.requests.post",
                    return_value=MockResponse({"response": "irrelevant"}, status_code=503),
                ):
                    result = semantic_judge("anything")

    assert result == "DANGEROUS"
    rec_fail.assert_called_once()
