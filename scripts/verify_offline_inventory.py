"""Read-only model/artifact fingerprints; no runtime imports, downloads or inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(output, record):
    """Replace only this invocation's already-reserved receipt atomically."""
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent,
                                     prefix='.inventory-', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(record, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path, cache, deadline, budget):
    actual = path.resolve(strict=True)
    if not actual.is_relative_to(cache.resolve()):
        raise ValueError('Model cache link escapes the declared cache root')
    before = actual.stat()
    if before.st_size > budget[0]:
        raise ValueError('Inventory byte budget exhausted')
    digest = hashlib.sha256()
    with actual.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError('Inventory deadline exhausted')
            digest.update(chunk)
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise ValueError('Inventory byte budget exhausted')
    after = actual.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Model file changed during fingerprinting')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def inspect_model(entry, cache, deadline, budget):
    model_id, revision = entry['model_id'], entry['revision']
    if (not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*', model_id)
            or not re.fullmatch(r'[0-9a-f]{40}', revision)):
        raise ValueError('Model identity must be a repository and a pinned 40-character commit')
    snapshot = cache / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
    filename = entry.get('filename', 'model.safetensors')
    if Path(filename).name != filename or filename in ('.', '..'):
        raise ValueError('Weight filename must be a single safe path component')
    if not snapshot.is_dir() or not (snapshot / filename).is_file():
        return {'model_id': model_id, 'revision': revision, 'status': 'MISSING'}
    weight = fingerprint(snapshot / filename, cache, deadline, budget)
    metadata = {}
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
                 'vocab.txt', 'spm.model', 'sentencepiece.bpe.model', 'modules.json',
                 '1_Pooling/config.json', 'sentence_bert_config.json', 'config_sentence_transformers.json'):
        if (snapshot / name).is_file():
            metadata[name] = fingerprint(snapshot / name, cache, deadline, budget)
    expected = entry.get('sha256')
    status = ('PASS' if weight['sha256'] == expected else 'FAIL') if expected else 'FINGERPRINTED_UNVERIFIED'
    return {'model_id': model_id, 'revision': revision, 'status': status,
            'weight': weight, 'expected_sha256': expected, 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=300)
    parser.add_argument('--max-bytes', type=int, default=16 * 1024 ** 3)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.max_bytes <= 0:
        parser.error('Time and byte budgets must be positive')
    output = args.output.resolve()
    profile = json.loads((ROOT / 'config/canonical_replay.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'models/detector_manifest.json').read_text(encoding='utf-8'))
    entries = dict(profile['detectors']) | {'embedding': profile['embedding'],
                                          'native_judge': manifest['llama_guard_3_1b']}
    record = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Read-only file identities; no model loading, inference or service acceptance',
              'cache': str(args.cache.resolve()), 'models': {}, 'artifacts': {}, 'runtime': {},
              'limitations': ['Embedding digest is observed, not checked against an independent pin.',
                              'Ollama served model identity and live judge remain unverified.',
                              'Metadata hashes are observed fingerprints, not independent approval.',
                              'No claim about model quality or readiness follows from file hashes.']}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    deadline, budget = time.monotonic() + args.timeout_seconds, [args.max_bytes]
    try:
        for name, entry in entries.items():
            result = inspect_model(entry, args.cache, deadline, budget)
            record['models'][name] = result
            checkpoint(output, record)
            print(name + ': ' + result['status'], flush=True)
        for name, expected in profile['artifacts'].items():
            path = ROOT / name
            actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if path.is_file() else None
            record['artifacts'][name] = {'expected': expected, 'actual': actual,
                                         'status': 'PASS' if actual == expected else 'FAIL'}
        for name, expected in profile['runtime_reference'].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            record['runtime'][name] = {'expected': expected, 'actual': actual,
                                       'status': 'PASS' if actual == expected else 'FAIL'}
        failed = any(result['status'] in ('FAIL', 'MISSING') for result in record['models'].values())
        failed |= any(result['status'] != 'PASS' for result in record['artifacts'].values())
        failed |= any(result['status'] != 'PASS' for result in record['runtime'].values())
        record['status'] = 'FAIL' if failed else 'PASS_WITH_LIMITATIONS'
    except BaseException as exc:
        record['status'] = 'INCOMPLETE'
        record['failure_type'] = type(exc).__name__
        raise
    finally:
        record['bytes_read'] = args.max_bytes - budget[0]
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint(output, record)
    print(record['status'], flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
