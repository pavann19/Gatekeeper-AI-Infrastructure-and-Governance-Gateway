"""Portable source verification using owned temporary state."""
import argparse
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
    args = parser.parse_args()
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
        for key in ('REDIS_URL', 'TENANTS_FILE', 'API_KEYS_FILE', 'AUDIT_LOG_PATH',
                    'AUDIT_DB_PATH', 'REVIEW_QUEUE_FILE', 'POLICY_RULES_FILE', 'POLICY_VERSIONS_DIR',
                    'WARM_MODELS_ON_STARTUP'):
            env.pop(key, None)
        env['PYTHONPATH'] = str(workspace)
        # The owned working directory contains default-path state. Environment
        # overrides would invalidate tests that assert the public defaults.
        (workspace / 'tenants.json').write_text('{}', encoding='utf-8')
        (workspace / 'api_keys.json').write_text('{}', encoding='utf-8')
        stages = [
            ('lint', ['-m', 'ruff', 'check', 'core/', 'api/']),
            ('types', ['-m', 'mypy', 'core/', 'api/', '--ignore-missing-imports', '--explicit-package-bases']),
            ('tests', ['-m', 'pytest', '-o', 'addopts=', '-m', 'not slow' if args.profile == 'unit' else '',
                       '--cov=core', '--cov=api', '--cov-fail-under=65',
                       '--junitxml=' + str(output / 'tests.xml')]),
        ]
        if args.profile == 'release':
            stages.append(('canonical', ['-m', 'scripts.canonical_replay', '--output', str(output / 'canonical')]))
        try:
            for name, command in stages:
                print('Verifying ' + name, flush=True)
                with (output / (name + '.log')).open('w', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, *command], cwd=workspace, env=env,
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
