"""
Live verification of the Redis-backed distributed rate limiter, token quota
tracker, and exact-match cache against a REAL running Redis instance -- not
mocks, not a stateful simulation. This is the one item this project's own
docs/RELEASE_CHECKLIST.md left explicitly unchecked.

Requires: a real Redis reachable at REDIS_URL (default redis://localhost:6379/0),
and this project's own core/* modules importable (run from the repo root).

Usage:
    REDIS_URL=redis://localhost:6379/0 python -m scripts.live_redis_verification
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import time

results = []


def record(name, passed, detail=""):
    results.append({"test": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not passed else ""))


def test_real_connection_and_pool_sharing():
    from core.redis_client import get_redis_client, reset_redis_client

    reset_redis_client()
    client = get_redis_client()
    record("real_redis_client_obtained", client is not None, "get_redis_client() returned None -- REDIS_URL unreachable?")
    if client is None:
        return False

    pong = client.ping()
    record("real_redis_ping_succeeds", pong is True, f"ping returned {pong!r}")

    from core.cache_backend import build_exact_cache_backend
    from core.rate_limit import build_rate_limiter
    from core.token_quota import build_token_quota_tracker

    cache = build_exact_cache_backend()
    limiter_a = build_rate_limiter("verify-a")
    limiter_b = build_rate_limiter("verify-b")
    quota = build_token_quota_tracker()

    same_client = (
        cache._client is client
        and limiter_a._client is client
        and limiter_b._client is client
        and quota._client is client
    )
    record("all_four_subsystems_share_one_real_client", same_client,
           "at least one subsystem opened its own separate connection instead of reusing the pool")
    return True


def test_rate_limiter_real_token_bucket_math():
    from core.rate_limit import build_rate_limiter

    limiter = build_rate_limiter("verify-bucket")
    limiter.reset()
    identity = "live-verify-user"

    # Burst of exactly 3 from capacity 3 must all succeed.
    outcomes = [limiter.check(identity, capacity=3.0, refill_per_second=1.0) for _ in range(3)]
    record("real_redis_burst_of_capacity_all_allowed", all(a for a, _ in outcomes), f"{outcomes}")

    # The 4th immediate request must be denied.
    allowed, retry_after = limiter.check(identity, capacity=3.0, refill_per_second=1.0)
    record("real_redis_over_capacity_denied", not allowed, f"got allowed={allowed}")
    record("real_redis_retry_after_is_positive", retry_after > 0, f"retry_after={retry_after}")

    # After waiting past the refill window, one more token must be available.
    time.sleep(1.1)
    allowed2, _ = limiter.check(identity, capacity=3.0, refill_per_second=1.0)
    record("real_redis_refill_after_real_wait_allows_again", allowed2, f"got allowed={allowed2}")

    limiter.reset()


def test_rate_limiter_real_concurrent_multiprocess_like_access():
    """
    The actual gap this whole verification pass exists for: does the Lua
    script's atomicity genuinely hold under REAL concurrent access from
    multiple independent connections against REAL Redis (not asyncio
    cooperative concurrency against a mock, which never exercises Redis's
    actual server-side atomicity guarantee at all)?
    """
    from core.rate_limit import build_rate_limiter

    limiter = build_rate_limiter("verify-concurrent")
    limiter.reset()
    identity = "concurrent-user"
    capacity = 50.0

    def one_check(_):
        return limiter.check(identity, capacity=capacity, refill_per_second=0.001)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        outcomes = list(ex.map(one_check, range(200)))

    allowed_count = sum(1 for a, _ in outcomes if a)
    record(
        "real_redis_concurrent_access_never_overspends_capacity",
        allowed_count == int(capacity),
        f"expected exactly {int(capacity)} allowed out of 200 concurrent requests "
        f"(capacity={capacity}, negligible refill), got {allowed_count} -- "
        f"a race in the atomic Lua script would show as MORE than capacity allowed",
    )
    limiter.reset()


def test_token_quota_real_atomic_increment_under_concurrency():
    from core.token_quota import build_token_quota_tracker

    tracker = build_token_quota_tracker()
    tracker.reset()
    tenant = "live-verify-tenant"

    def one_record(_):
        tracker.record(tenant, 10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(one_record, range(100)))

    total = tracker.usage_today(tenant)
    record(
        "real_redis_concurrent_token_increments_sum_correctly",
        total == 1000,
        f"expected exactly 1000 (100 concurrent increments of 10 each), got {total} -- "
        f"a lost update from a non-atomic increment would show as LESS than 1000",
    )
    tracker.reset()


def test_cache_real_round_trip_and_flush():
    from core.cache_backend import build_exact_cache_backend

    cache = build_exact_cache_backend()
    cache.flush()

    cache.set("live-verify-key", {"decision": "BLOCK", "risk": "HIGH"}, ttl_seconds=60)
    got = cache.get("live-verify-key")
    record("real_redis_cache_round_trip", got == {"decision": "BLOCK", "risk": "HIGH"}, f"got {got}")

    cache.delete("live-verify-key")
    got_after_delete = cache.get("live-verify-key")
    record("real_redis_cache_delete_removes_entry", got_after_delete is None, f"got {got_after_delete}")

    cache.flush()


def test_graceful_fallback_when_redis_becomes_unreachable():
    """
    Simulates the operational scenario docs/THREAT_MODEL.md §8 describes:
    Redis was reachable at startup, then becomes unreachable mid-process
    (network partition, Redis restart). The already-built RedisRateLimiter
    must fall back to its internal LocalRateLimiter rather than raise, or
    convert an availability blip into a 500 for every caller.
    """
    from core.rate_limit import build_rate_limiter

    limiter = build_rate_limiter("verify-fallback")
    # Point the SAME limiter instance's client at an address nothing is
    # listening on, simulating Redis going away without restarting our process.
    import redis as redis_module
    broken_client = redis_module.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=1, socket_timeout=1)
    limiter._client = broken_client
    limiter._script = None  # force the eval() fallback path to also hit the broken client

    allowed, retry_after = limiter.check("fallback-user", capacity=5.0, refill_per_second=1.0)
    record(
        "redis_becoming_unreachable_falls_back_to_local_not_a_crash",
        allowed is True,
        f"expected a graceful local-fallback allow, got allowed={allowed}",
    )


def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    print(f"=== Live Redis verification against {redis_url} ===")

    ok = test_real_connection_and_pool_sharing()
    if not ok:
        print("\nCannot reach real Redis -- aborting remaining tests.")
        sys.exit(1)

    test_rate_limiter_real_token_bucket_math()
    test_rate_limiter_real_concurrent_multiprocess_like_access()
    test_token_quota_real_atomic_increment_under_concurrency()
    test_cache_real_round_trip_and_flush()
    test_graceful_fallback_when_redis_becomes_unreachable()

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = [r for r in results if not r["passed"]]

    print(f"\n=== {passed}/{total} passed ===")
    if failed:
        print("FAILURES:")
        for f in failed:
            print(f"  - {f['test']}: {f['detail']}")

    import json
    with open("_evidence/live_redis_verification_results.json", "w", encoding="utf-8") as fh:
        json.dump({"redis_url_host": redis_url.split("@")[-1], "total": total, "passed": passed,
                   "failed": len(failed), "results": results}, fh, indent=2)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
