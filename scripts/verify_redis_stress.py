"""Bounded multi-process Redis contracts using an owned loopback container.

This is backend correctness evidence, not API capacity, billing enforcement,
application replica acceptance, or detector quality. No model or provider runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

from scripts.verify_redis import IMAGE, wait_ready

_barrier = None


def initialize(barrier):
    global _barrier
    _barrier = barrier


def compete(url, prefix, attempts):
    import redis
    from core.cache_backend import RedisExactCache
    from core.rate_limit import RedisRateLimiter
    from core.token_quota import RedisTokenQuotaTracker

    client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
    cache, limiter, quota = RedisExactCache(client), RedisRateLimiter(client, "stress"), RedisTokenQuotaTracker(client)
    cache.KEY_PREFIX, limiter.KEY_PREFIX, quota.KEY_PREFIX = prefix + 'cache:', prefix + 'rate:', prefix + 'quota:'
    try:
        _barrier.wait(timeout=45)
        high = admitted = 0
        for _ in range(attempts):
            high += cache.set('protected', {'risk': 'LOW'}, 300)['risk'] == 'HIGH'
            admitted += limiter.check('tenant', 5, 0.000001, now=100)[0]
            quota.record('tenant', 2)
        return {'pid': os.getpid(), 'high_preserved': high, 'admitted': admitted, 'quota_writes': attempts}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--attempts-per-worker', type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.attempts_per_worker <= 10000:
        parser.error('Use 1..16 workers and 1..10000 attempts per worker')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'workers': args.workers, 'attempts_per_worker': args.attempts_per_worker,
              'image': IMAGE, 'scope': __doc__, 'source_sha256': {}}
    root = Path(__file__).resolve().parents[1]
    for name in ('core/cache_backend.py', 'core/rate_limit.py', 'core/token_quota.py', 'scripts/verify_redis_stress.py'):
        report['source_sha256'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    identity = None
    started = time.perf_counter()
    try:
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        created = subprocess.run(['docker', 'run', '--pull=never', '-d', '--name', 'gatekeeper-stress-' + uuid.uuid4().hex,
                                  '-p', f'127.0.0.1:{port}:6379', '--memory=128m', '--cpus=1',
                                  IMAGE, 'redis-server', '--save', '', '--appendonly', 'no'],
                                 capture_output=True, text=True, timeout=30)
        if created.returncode:
            raise RuntimeError('Owned container creation failed: ' + created.stderr)
        identity = created.stdout.strip()
        import redis
        from core.cache_backend import RedisExactCache
        from core.rate_limit import RedisRateLimiter
        from core.token_quota import RedisTokenQuotaTracker

        url, prefix = f'redis://127.0.0.1:{port}/0', 'gatekeeper:stress:' + uuid.uuid4().hex + ':'
        with redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2) as client:
            wait_ready(client)
            cache = RedisExactCache(client)
            cache.KEY_PREFIX = prefix + 'cache:'
            cache.set('protected', {'risk': 'HIGH'}, 300)
            context = multiprocessing.get_context('spawn')
            barrier = context.Barrier(args.workers)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                     initializer=initialize, initargs=(barrier,)) as pool:
                futures = [pool.submit(compete, url, prefix, args.attempts_per_worker) for _ in range(args.workers)]
                results = [future.result(timeout=180) for future in futures]
            total = args.workers * args.attempts_per_worker
            quota = RedisTokenQuotaTracker(client)
            quota.KEY_PREFIX = prefix + 'quota:'
            report.update(results=results, attempts=total, high_preserved=sum(r['high_preserved'] for r in results),
                          admitted=sum(r['admitted'] for r in results), recorded_tokens=quota.usage_today('tenant'),
                          distinct_processes=len({r['pid'] for r in results}))
            assert report['distinct_processes'] == args.workers
            assert report['high_preserved'] == total and cache.get('protected')['risk'] == 'HIGH'
            assert report['admitted'] == 5 and report['recorded_tokens'] == total * 2
            # A fresh identity has its own budget; tenant usage cannot bleed.
            quota.record('other', 7)
            assert quota.usage_today('other') == 7 and quota.usage_today('tenant') == total * 2
            limiter = RedisRateLimiter(client, 'downgrade')
            limiter.KEY_PREFIX = prefix + 'rate:'
            assert limiter.check('tenant', 100, 1, now=100)[0]
            report['admitted_after_capacity_reduction'] = sum(limiter.check('tenant', 5, 1, now=100)[0] for _ in range(100))
            assert report['admitted_after_capacity_reduction'] == 5
            report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        if identity:
            removed = subprocess.run(['docker', 'rm', '-f', identity], capture_output=True, text=True, timeout=30)
            report['owned_container_removed'] = removed.returncode == 0
        report['elapsed_seconds'] = time.perf_counter() - started
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
