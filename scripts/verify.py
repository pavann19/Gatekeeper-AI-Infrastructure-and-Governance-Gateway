"""Portable source verification using owned temporary state."""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('unit', 'release'), default='release')
    parser.add_argument('--canonical-model-cache', type=Path,
                        help='Explicit provisioned Hugging Face hub cache for the release-only frozen replay')
    args = parser.parse_args()
    if args.profile == 'release' and (
        args.canonical_model_cache is None or not args.canonical_model_cache.is_dir()
    ):
        parser.error('Release replay requires an existing --canonical-model-cache; source tests use an empty cache')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    results = []
    with tempfile.TemporaryDirectory(prefix='gatekeeper-verify-') as temporary:
        workspace = Path(temporary) / 'source'
        workspace.mkdir()
        for name in files:
            if not name:
                continue
            source = root / name
            if source.is_symlink():
                raise RuntimeError('Symlink source is outside verifier contract')
            if source.is_file():
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        env = os.environ.copy()
        # Discover every settings override without importing application code.
        # A fixed subset omitted DATABASE_URL/STATE_BACKEND and could connect
        # an apparently isolated source check to inherited shared resources.
        configuration = ast.parse((workspace / 'core/config.py').read_text(encoding='utf-8'))
        for node in ast.walk(configuration):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                env.pop(node.target.id, None)
        for key in ('REDIS_URL', 'GATEKEEPER_REDIS_INTEGRATION_URL', 'GATEKEEPER_TEST_PG_URL'):
            env.pop(key, None)
        env['PYTHONPATH'] = str(workspace)
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            env[key] = '1'
        for key, directory in (('HF_HOME', 'hf'), ('HF_HUB_CACHE', 'hf/hub'),
                               ('SENTENCE_TRANSFORMERS_HOME', 'sentence-transformers'), ('TORCH_HOME', 'torch')):
            env[key] = str(Path(temporary) / 'empty-model-cache' / directory)
        # The owned working directory contains default-path state. Environment
        # overrides would invalidate tests that assert the public defaults.
        # Missing optional store permits only the literal default tenant.
        # A present empty object denies every tenant, masking route contracts.
        (workspace / 'api_keys.json').write_text('{}', encoding='utf-8')
        stages = [
            ('lint', ['-m', 'ruff', 'check', 'core/', 'api/']),
            ('types', ['-m', 'mypy', 'core/', 'api/', '--ignore-missing-imports', '--explicit-package-bases']),
            ('tests', ['-m', 'pytest', '-o', 'addopts=', '-m', 'not slow and not integration', '--timeout=60',
                       '--cov=core', '--cov=api', '--cov-fail-under=65',
                       '--junitxml=' + str(output / 'tests.xml')]),
        ]
        if args.profile == 'release':
            stages.append(('canonical', ['-m', 'scripts.canonical_replay', '--output', str(output / 'canonical')]))
        try:
            for name, command in stages:
                print('Verifying ' + name, flush=True)
                stage_env = env.copy()
                if name == 'canonical':
                    # Source tests never read real weights. Frozen replay is a
                    # separate explicit, offline, provisioned-model contract.
                    for key in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
                                'SENTENCE_TRANSFORMERS_HOME'):
                        stage_env[key] = str(args.canonical_model_cache.resolve())
                with (output / (name + '.log')).open('w', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, *command], cwd=workspace, env=stage_env,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                results.append({'stage': name, 'exit_code': result.returncode})
                if result.returncode:
                    raise RuntimeError(name + ' failed; see ' + str(output / (name + '.log')))
        finally:
            (output / 'verification.json').write_text(json.dumps({
                'profile': args.profile, 'stages': results, 'release_ready': False,
                'scope': 'Source checks only; deployment, Redis and live benchmark acceptance are separate.'
            }, indent=2), encoding='utf-8')
    print('Source verification passed for profile ' + args.profile)


if __name__ == '__main__':
    main()
