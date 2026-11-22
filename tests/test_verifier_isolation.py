"""The source verifier must never inherit authority over real services/state."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify


def test_verifier_removes_all_setting_authority_and_excludes_service_tests(tmp_path, monkeypatch):
    root = tmp_path / 'synthetic-repository'
    (root / 'scripts').mkdir(parents=True)
    (root / 'core').mkdir()
    (root / 'core/config.py').write_text('''
class Settings:
    STATE_BACKEND: str = "local"
    DATABASE_URL: str = ""
    TENANTS_FILE: str = "tenants.json"
    CACHE_EXACT_FILE: str = "cache.json"
    OPENAI_API_KEY: str = ""
''', encoding='utf-8')
    monkeypatch.setattr(verify, '__file__', str(root / 'scripts/verify.py'))
    monkeypatch.setattr(verify.subprocess, 'check_output', lambda *a, **kw: b'core/config.py\0')
    for name, value in {'STATE_BACKEND': 'postgres', 'DATABASE_URL': 'synthetic-hostile-db',
                        'TENANTS_FILE': 'synthetic-hostile-tenant-store', 'CACHE_EXACT_FILE': 'synthetic-hostile-cache',
                        'OPENAI_API_KEY': 'synthetic-hostile-token', 'REDIS_URL': 'synthetic-hostile-redis',
                        'GATEKEEPER_TEST_PG_URL': 'synthetic-hostile-pg',
                        'GATEKEEPER_REDIS_INTEGRATION_URL': 'synthetic-hostile-redis',
                        'HF_HOME': 'synthetic-hostile-model-cache'}.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / 'receipts'
    monkeypatch.setattr(verify.sys, 'argv', ['verify', '--profile', 'unit', '--output', str(output)])
    stages = []

    def inspect(command, cwd, env, **kwargs):
        assert not any('synthetic-hostile' in value for value in env.values())
        assert not (Path(cwd) / 'tenants.json').exists()
        for name in ('STATE_BACKEND', 'DATABASE_URL', 'REDIS_URL', 'OPENAI_API_KEY', 'GATEKEEPER_TEST_PG_URL'):
            assert name not in env
        for name in ('HF_HOME', 'HF_HUB_CACHE', 'SENTENCE_TRANSFORMERS_HOME', 'TORCH_HOME'):
            assert Path(env[name]).is_relative_to(Path(cwd).parent)
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            assert env[name] == '1'
        if 'pytest' in command:
            assert command[command.index('-m', 3) + 1] == 'not slow and not integration'
            assert '--timeout=60' in command
        stages.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verify.subprocess, 'run', inspect)
    verify.main()
    assert len(stages) == 3
    assert json.loads((output / 'verification.json').read_text())['release_ready'] is False


def test_release_requires_explicit_existing_model_cache_before_creating_output(tmp_path, monkeypatch):
    output = tmp_path / 'must-not-be-created'
    monkeypatch.setattr(verify.sys, 'argv', ['verify', '--profile', 'release', '--output', str(output)])
    with pytest.raises(SystemExit) as failure:
        verify.main()
    assert failure.value.code == 2
    assert not output.exists()
