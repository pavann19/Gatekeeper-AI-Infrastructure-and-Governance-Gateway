"""Run candidate tests in a fresh source copy with empty local state and model caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path,
                        help='Copy only the pinned embedding snapshot from this read-only HF hub cache')
    parser.add_argument('--integration', action='store_true')
    parser.add_argument('--postgres-fixture', action='store_true',
                        help='Use an explicitly provided owned loopback synthetic PostgreSQL fixture')
    parser.add_argument('--redis-fixture', action='store_true')
    parser.add_argument('--redis-outage-fixture', action='store_true')
    parser.add_argument('tests', nargs='+')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    source = run / 'source'
    source.mkdir()
    names = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'],
                           capture_output=True, check=True).stdout.decode().split('\0')
    # Include new candidate code/tests, but never runtime state or historical evidence.
    for directory in ('core', 'api', 'tests', 'evaluation', 'scripts'):
        names.extend(str(p.relative_to(ROOT)).replace('\\', '/')
                     for p in (ROOT / directory).rglob('*.py'))
    names.extend(str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'deploy').rglob('*.yml'))
    names.extend(str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'migrations').rglob('*.sql'))
    hashes = {}
    for name in sorted(set(names)):
        path = ROOT / name
        if (not name or not path.is_file() or path.is_symlink()
                or any(part.startswith('.') for part in Path(name).parts)
                or name.startswith(('evidence/', '_evidence/'))):
            continue
        if path.suffix not in {'.py', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.html', '.js', '.css', '.txt', '.sql'}:
            continue
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    model_sources = {}
    if args.embedding_cache:
        cache = args.embedding_cache.resolve(strict=True)
        profile = json.loads((ROOT / 'config/canonical_replay.json').read_text())['embedding']
        relative = Path('models--' + profile['model_id'].replace('/', '--')) / 'snapshots' / profile['revision']
        snapshot = cache / relative
        if not snapshot.is_dir():
            raise RuntimeError('Pinned embedding snapshot is missing; downloads are forbidden')
        total = 0
        for path in snapshot.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json', '.txt', '.safetensors', '.model'):
                continue
            if not path.resolve().is_relative_to(cache):
                raise RuntimeError('Embedding snapshot link escapes the declared cache')
            total += path.stat().st_size
            if total > 2 * 1024 ** 3:
                raise RuntimeError('Embedding fixture exceeds its 2 GiB copy budget')
            target = run / 'empty-hf-cache' / 'hub' / relative / path.relative_to(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            model_sources[str(path)] = file_sha(path)
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP',
            'USERPROFILE', 'APPDATA', 'LOCALAPPDATA') if key in os.environ}
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(source),
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'USE_TF': '0',
                'PYTHONIOENCODING': 'utf-8',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'HF_HOME': str(run / 'empty-hf-cache'), 'SENTENCE_TRANSFORMERS_HOME': str(run / 'empty-st-cache'),
                'WARM_MODELS_ON_STARTUP': 'false', 'AUTH_MODE': 'optional',
                'OLLAMA_API_URL': 'http://127.0.0.1:1/api/generate',
                'OLLAMA_CHAT_URL': 'http://127.0.0.1:1/api/chat'})
    if args.embedding_cache:
        # SentenceTransformer passes its own cache folder to Hub APIs. It must
        # point at the copied snapshot, rather than a second empty cache.
        env['SENTENCE_TRANSFORMERS_HOME'] = str(run / 'empty-hf-cache' / 'hub')
    if args.postgres_fixture:
        import psycopg2.extensions
        dsn = os.environ['GATEKEEPER_TEST_PG_URL']
        parsed = psycopg2.extensions.parse_dsn(dsn)
        if parsed.get('host') != '127.0.0.1' or not parsed.get('dbname', '').startswith('gk_synthetic_'):
            raise RuntimeError('Only owned loopback synthetic PostgreSQL fixtures are admitted')
        env.update(STATE_BACKEND='postgres', DATABASE_URL=dsn, GATEKEEPER_TEST_PG_URL=dsn)
    if args.redis_fixture or args.redis_outage_fixture:
        from urllib.parse import urlsplit
        url = os.environ['GATEKEEPER_REDIS_INTEGRATION_URL']
        parsed = urlsplit(url)
        if parsed.scheme != 'redis' or parsed.hostname != '127.0.0.1':
            raise RuntimeError('Only owned loopback Redis fixtures are admitted')
        env['GATEKEEPER_REDIS_INTEGRATION_URL'] = url
        if args.redis_outage_fixture:
            env.update(REDIS_URL=url, GATEKEEPER_REDIS_OUTAGE_FIXTURE='true')
    command = [sys.executable, '-B', '-m', 'pytest', '-p', 'pytest_timeout', '-p', 'pytest_asyncio.plugin', '--timeout=60',
               '-p', 'no:cacheprovider', '--junitxml=' + str(run / 'pytest.xml'), *args.tests]
    if args.integration:
        command.extend(['-m', 'integration'])
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
              'scope': 'Copied source; empty local state/model caches; no inherited service credentials',
              'source_sha256': hashes,
              'environment': {key: ('[redacted fixture DSN]' if key in (
                                    'DATABASE_URL', 'GATEKEEPER_TEST_PG_URL', 'REDIS_URL', 'GATEKEEPER_REDIS_INTEGRATION_URL')
                                    else '[host executable path]' if key == 'PATH' else value)
                              for key, value in env.items()},
              'model_fixture_source_sha256': model_sources,
              'result': 'RUNNING'}
    receipt = run / 'receipt.json'
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    with (run / 'pytest.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=900)
            record['exit_code'] = result.returncode
            record['result'] = 'PASS' if result.returncode == 0 else 'FAIL'
        except subprocess.TimeoutExpired:
            record['result'] = 'TIMEOUT'
            record['exit_code'] = 124
        finally:
            record['finished_utc'] = datetime.now(timezone.utc).isoformat()
            record['source_drift'] = [name for name, digest in hashes.items()
                                      if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
            record['model_fixture_source_drift'] = [name for name, digest in model_sources.items()
                                                   if file_sha(Path(name)) != digest]
            receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'run': str(run), 'result': record['result'], 'source_drift': record['source_drift']}))
    print((run / 'pytest.log').read_text(encoding='utf-8', errors='replace')[-14000:])
    return record['exit_code'] or int(bool(record['source_drift'] or record['model_fixture_source_drift']))


if __name__ == '__main__':
    raise SystemExit(main())
