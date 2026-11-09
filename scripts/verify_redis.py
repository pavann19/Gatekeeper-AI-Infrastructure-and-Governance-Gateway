"""Verify real Redis using an owned container and private key namespace."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import redis

IMAGE = 'redis@sha256:858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499'


def wait_ready(client):
    for _ in range(40):
        try:
            if client.ping():
                return
        except redis.RedisError:
            time.sleep(0.2)
    raise RuntimeError('Owned Redis did not become ready')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    # Explicit host port survives container stop/start. A competing bind causes
    # docker run to fail; no existing listener is terminated or reused.
    created = subprocess.run(['docker', 'run', '-d', '--name', 'gatekeeper-verify-' + uuid.uuid4().hex,
                              '-p', f'127.0.0.1:{port}:6379', IMAGE, 'redis-server', '--appendonly', 'yes',
                              '--appendfsync', 'always'], capture_output=True, text=True, timeout=180)
    if created.returncode:
        raise RuntimeError(created.stderr)
    identity = created.stdout.strip()
    try:
        url = f'redis://127.0.0.1:{port}/0'
        client = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        wait_ready(client)
        env = {**os.environ, 'GATEKEEPER_REDIS_INTEGRATION_URL': url}
        with (output / 'concurrency.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_live_redis_acceptance.py', '-q'],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError('Redis concurrency acceptance failed')
        from core.cache_backend import RedisExactCache
        from core.rate_limit import RedisRateLimiter
        from core.token_quota import RedisTokenQuotaTracker
        prefix = 'gatekeeper:owned:' + uuid.uuid4().hex + ':'
        cache = RedisExactCache(client)
        limiter = RedisRateLimiter(client, 'restart')
        quota = RedisTokenQuotaTracker(client)
        cache.KEY_PREFIX, limiter.KEY_PREFIX, quota.KEY_PREFIX = prefix + 'cache:', prefix + 'rate:', prefix + 'quota:'
        cache.set('key', {'risk': 'HIGH'}, 60)
        assert limiter.check('owner', 1, 0.000001, now=100)[0]
        quota.record('tenant', 100)
        subprocess.run(['docker', 'stop', identity], check=True, capture_output=True, timeout=30)
        assert not limiter.check('owner', 1, 1)[0]
        assert quota.would_exceed('tenant', 100)
        try:
            quota.record('tenant', 1)
        except ConnectionError:
            pass
        else:
            raise AssertionError('Unavailable quota persistence was accepted')
        subprocess.run(['docker', 'start', identity], check=True, capture_output=True, timeout=30)
        wait_ready(client)
        assert cache.get('key')['risk'] == 'HIGH'
        assert not limiter.check('owner', 1, 0.000001, now=100)[0]
        assert quota.usage_today('tenant') == 100
        (output / 'result.json').write_text(json.dumps({
            'concurrency': 'pass', 'outage': 'fail_closed', 'restart': 'pass', 'image': IMAGE,
            'scope': 'Independent backend clients; full application replica acceptance is separate'
        }, indent=2), encoding='utf-8')
        print('Redis concurrency, outage and persistent restart passed')
    finally:
        subprocess.run(['docker', 'rm', '-f', identity], capture_output=True, check=True, timeout=30)


if __name__ == '__main__':
    main()
