"""
Tests for core/config.py's Settings: defaults, env-var overrides, and
field validators. Settings is instantiated directly per test (never via
the module-level `settings` singleton) so env vars set with monkeypatch
are guaranteed to be picked up, and pydantic-settings' own .env loading
is disabled via _env_file=None so a real .env on disk can never leak in.
"""
import pytest
from pydantic import ValidationError

from core.config import Settings


def make_settings(**env):
    """Build a Settings instance from explicit env vars only, isolated
    from any real .env file and from the actual process environment."""
    return Settings(_env_file=None, **env)


# --- Defaults ----------------------------------------------------------

def test_capability_tier_defaults():
    s = make_settings()
    assert s.CAPABILITY_GENERAL == "GENERAL"
    assert s.CAPABILITY_ELEVATED == "ELEVATED"
    assert s.CAPABILITY_INTERNAL == "INTERNAL"


def test_calibrated_threshold_defaults():
    s = make_settings()
    assert s.SEMANTIC_THRESHOLD_HIGH == 0.48
    assert s.SEMANTIC_THRESHOLD_MEDIUM == 0.30
    assert s.META_INTENT_THRESHOLD == 0.30
    assert s.EDUCATIONAL_THRESHOLD == 0.45
    assert s.DOMAIN_THRESHOLD == 0.22
    assert s.CACHE_SIMILARITY_THRESHOLD == 0.99


def test_fusion_flag_defaults():
    s = make_settings()
    assert s.FUSION_PARALLEL is True
    assert s.FUSION_PER_CLASS is True
    assert s.JUDGE_MAY_CLEAR_TO_LOW is False


def test_domain_guardrail_defaults_to_off():
    s = make_settings()
    assert s.DOMAIN_GUARDRAIL_MODE == "off"


def test_execution_environment_defaults():
    s = make_settings()
    assert s.OLLAMA_API_URL == "http://localhost:11434/api/generate"
    assert s.OLLAMA_MODEL == "llama-guard3"  # not the stale "mistral" default
    assert s.EMBEDDING_MODEL == "all-mpnet-base-v2"


def test_llm_provider_defaults_are_empty_not_missing(monkeypatch):
    # The real shell environment may itself define ANTHROPIC_BASE_URL (it
    # does in this sandbox) -- clear it so this test reflects the *default*
    # baked into Settings, not whatever happens to be in the process env.
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    s = make_settings()
    assert s.OLLAMA_CHAT_URL == "http://localhost:11434/api/chat"
    assert s.OPENAI_API_KEY == ""
    assert s.OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert s.OPENAI_MODEL == ""
    assert s.ANTHROPIC_API_KEY == ""
    assert s.ANTHROPIC_BASE_URL == "https://api.anthropic.com/v1"
    assert s.ANTHROPIC_MODEL == ""
    assert s.ANTHROPIC_VERSION == "2023-06-01"


def test_file_path_defaults():
    s = make_settings()
    assert s.POLICY_FILE == "policies.json"
    assert s.POLICY_RULES_FILE == "policy_rules.json"
    assert s.POLICY_VERSIONS_DIR == "policy_versions"
    assert s.EVIDENCE_DIR == "_evidence"
    assert s.REVIEW_QUEUE_FILE == "review_queue.json"
    assert s.AUDIT_LOG_PATH == "audit.jsonl"


def test_auth_defaults_to_optional_least_privilege():
    s = make_settings()
    assert s.AUTH_MODE == "optional"
    assert s.API_KEYS_FILE == "api_keys.json"
    assert s.TENANTS_FILE == "tenants.json"


def test_cors_defaults_to_wildcard():
    s = make_settings()
    assert s.CORS_ORIGINS == "*"


def test_rate_limit_defaults():
    s = make_settings()
    assert s.RATE_LIMIT_ENABLED is True
    assert s.RATE_LIMIT_AUTHENTICATED_RPM == 120.0
    assert s.RATE_LIMIT_ANONYMOUS_RPM == 20.0
    assert s.RATE_LIMIT_BURST_SECONDS == 10.0
    assert s.RATE_LIMIT_MAX_TRACKED == 10_000
    assert s.RATE_LIMIT_TRUST_FORWARDED_FOR is False


def test_assess_execution_bound_defaults():
    s = make_settings()
    assert s.ASSESS_MAX_CONCURRENCY == 4
    assert s.ASSESS_TIMEOUT_SECONDS == 30.0


def test_gateway_execution_bound_defaults():
    s = make_settings()
    assert s.GATEWAY_MAX_CONCURRENCY == 4
    assert s.GATEWAY_TIMEOUT_SECONDS == 60.0
    assert s.LLM_GATEWAY_DEFAULT_PROVIDER == "ollama"
    assert s.GATEWAY_FALLBACK_PROVIDERS == ""


def test_token_quota_defaults_to_disabled_and_unlimited():
    s = make_settings()
    assert s.GATEWAY_TOKEN_QUOTA_ENABLED is False
    assert s.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT == 0


def test_warm_models_and_demo_tools_defaults():
    s = make_settings()
    assert s.WARM_MODELS_ON_STARTUP is True
    assert s.REGISTER_DEMO_TOOLS is False


def test_tool_http_get_allowlist_defaults_to_empty_fail_closed():
    s = make_settings()
    assert s.TOOL_HTTP_GET_ALLOWED_DOMAINS == ""


def test_metrics_defaults():
    s = make_settings()
    assert s.METRICS_ENABLED is True
    assert s.METRICS_PATH == "/metrics"
    assert s.METRICS_REQUIRE_AUTH is False


def test_request_id_defaults():
    s = make_settings()
    assert s.REQUEST_ID_HEADER == "X-Request-ID"
    assert s.REQUEST_ID_MAX_LENGTH == 64


def test_spacy_model_default():
    s = make_settings()
    assert s.SPACY_MODEL == "en_core_web_sm"


# --- Env-var overrides ---------------------------------------------------

def test_env_override_string_setting(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    s = make_settings()
    assert s.OLLAMA_MODEL == "custom-model"


def test_env_override_bool_setting(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    s = make_settings()
    assert s.RATE_LIMIT_ENABLED is False


def test_env_override_float_setting(monkeypatch):
    monkeypatch.setenv("SEMANTIC_THRESHOLD_HIGH", "0.75")
    s = make_settings()
    assert s.SEMANTIC_THRESHOLD_HIGH == 0.75


def test_env_override_int_setting(monkeypatch):
    monkeypatch.setenv("ASSESS_MAX_CONCURRENCY", "8")
    s = make_settings()
    assert s.ASSESS_MAX_CONCURRENCY == 8


def test_env_var_type_coercion_from_string(monkeypatch):
    # pydantic-settings reads all env vars as strings and coerces them
    monkeypatch.setenv("RATE_LIMIT_MAX_TRACKED", "500")
    s = make_settings()
    assert s.RATE_LIMIT_MAX_TRACKED == 500
    assert isinstance(s.RATE_LIMIT_MAX_TRACKED, int)


def test_invalid_int_env_var_raises():
    with pytest.raises(ValidationError):
        make_settings(ASSESS_MAX_CONCURRENCY="not-a-number")


def test_invalid_bool_env_var_raises():
    with pytest.raises(ValidationError):
        make_settings(RATE_LIMIT_ENABLED="not-a-bool")


# --- DOMAIN_GUARDRAIL_MODE validator --------------------------------------

@pytest.mark.parametrize("mode", ["off", "advisory", "enforcing"])
def test_domain_guardrail_mode_accepts_valid_values(mode):
    s = make_settings(DOMAIN_GUARDRAIL_MODE=mode)
    assert s.DOMAIN_GUARDRAIL_MODE == mode


def test_domain_guardrail_mode_normalises_case_and_whitespace():
    s = make_settings(DOMAIN_GUARDRAIL_MODE="  ENFORCING  ")
    assert s.DOMAIN_GUARDRAIL_MODE == "enforcing"


def test_domain_guardrail_mode_rejects_invalid_value():
    with pytest.raises(ValidationError, match="DOMAIN_GUARDRAIL_MODE"):
        make_settings(DOMAIN_GUARDRAIL_MODE="bogus")


# --- AUTH_MODE validator --------------------------------------------------

@pytest.mark.parametrize("mode", ["optional", "required"])
def test_auth_mode_accepts_valid_values(mode):
    s = make_settings(AUTH_MODE=mode)
    assert s.AUTH_MODE == mode


def test_auth_mode_normalises_case_and_whitespace():
    s = make_settings(AUTH_MODE="  REQUIRED  ")
    assert s.AUTH_MODE == "required"


def test_auth_mode_rejects_invalid_value():
    with pytest.raises(ValidationError, match="AUTH_MODE"):
        make_settings(AUTH_MODE="admin")


# --- RPM validator (rejects <=0, no "0 to disable" convention here) ------

@pytest.mark.parametrize("field", ["RATE_LIMIT_AUTHENTICATED_RPM", "RATE_LIMIT_ANONYMOUS_RPM"])
def test_rpm_rejects_zero(field):
    with pytest.raises(ValidationError, match="positive"):
        make_settings(**{field: 0})


@pytest.mark.parametrize("field", ["RATE_LIMIT_AUTHENTICATED_RPM", "RATE_LIMIT_ANONYMOUS_RPM"])
def test_rpm_rejects_negative(field):
    with pytest.raises(ValidationError, match="positive"):
        make_settings(**{field: -5})


def test_rpm_accepts_small_positive_boundary():
    s = make_settings(RATE_LIMIT_AUTHENTICATED_RPM=0.01, RATE_LIMIT_ANONYMOUS_RPM=0.01)
    assert s.RATE_LIMIT_AUTHENTICATED_RPM == 0.01


# --- Concurrency validators (must be >= 1) --------------------------------

def test_assess_max_concurrency_rejects_zero():
    with pytest.raises(ValidationError, match="at least 1"):
        make_settings(ASSESS_MAX_CONCURRENCY=0)


def test_assess_max_concurrency_accepts_boundary_one():
    s = make_settings(ASSESS_MAX_CONCURRENCY=1)
    assert s.ASSESS_MAX_CONCURRENCY == 1


def test_gateway_max_concurrency_rejects_negative():
    with pytest.raises(ValidationError, match="at least 1"):
        make_settings(GATEWAY_MAX_CONCURRENCY=-1)


def test_gateway_max_concurrency_accepts_boundary_one():
    s = make_settings(GATEWAY_MAX_CONCURRENCY=1)
    assert s.GATEWAY_MAX_CONCURRENCY == 1


# --- Timeout validators (must be > 0) -------------------------------------

def test_assess_timeout_rejects_zero():
    with pytest.raises(ValidationError, match="positive"):
        make_settings(ASSESS_TIMEOUT_SECONDS=0)


def test_assess_timeout_rejects_negative():
    with pytest.raises(ValidationError, match="positive"):
        make_settings(ASSESS_TIMEOUT_SECONDS=-1.0)


def test_gateway_timeout_rejects_zero():
    with pytest.raises(ValidationError, match="positive"):
        make_settings(GATEWAY_TIMEOUT_SECONDS=0)


def test_gateway_timeout_accepts_small_positive_boundary():
    s = make_settings(GATEWAY_TIMEOUT_SECONDS=0.001)
    assert s.GATEWAY_TIMEOUT_SECONDS == 0.001


# --- Token quota default validator (must be >= 0, 0 means unlimited) -----

def test_token_quota_default_rejects_negative():
    with pytest.raises(ValidationError, match=">= 0"):
        make_settings(GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT=-1)


def test_token_quota_default_accepts_zero_as_unlimited():
    s = make_settings(GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT=0)
    assert s.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT == 0


def test_token_quota_default_accepts_positive_value():
    s = make_settings(GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT=1000)
    assert s.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT == 1000


# --- .env loading behaviour ------------------------------------------------

def test_settings_loads_from_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OLLAMA_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Explicitly point at the tmp .env rather than relying on cwd + the
    # class-level default, so this never touches the real repo's .env.
    s = Settings(_env_file=str(env_path))
    assert s.OLLAMA_MODEL == "from-dotenv"


def test_explicit_env_var_overrides_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OLLAMA_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODEL", "from-process-env")
    s = Settings(_env_file=str(env_path))
    assert s.OLLAMA_MODEL == "from-process-env"


# --- Module-level backwards-compatibility exports -------------------------

def test_module_level_exports_match_settings_singleton():
    import core.config as config_mod
    assert config_mod.CAPABILITY_GENERAL == config_mod.settings.CAPABILITY_GENERAL
    assert config_mod.SEMANTIC_THRESHOLD_HIGH == config_mod.settings.SEMANTIC_THRESHOLD_HIGH
    assert config_mod.OLLAMA_MODEL == config_mod.settings.OLLAMA_MODEL
    assert config_mod.POLICY_VERSIONS_DIR == config_mod.settings.POLICY_VERSIONS_DIR
