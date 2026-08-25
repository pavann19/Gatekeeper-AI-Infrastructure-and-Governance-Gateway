# Deployment Runbook

This runbook records the release-layer decisions that are outside the
application code but must be true for an application release.

## Runtime Shape

The deployment environment supports both single-worker local execution and multi-replica distributed deployment:

- When `REDIS_URL` is set and reachable, `core/rate_limit.py` automatically initializes `RedisRateLimiter`, synchronizing token-bucket rate limits atomically across all API replicas via Redis server-clock Lua scripts.
- When `REDIS_URL` is not set (or during transient Redis outages), `core/rate_limit.py` gracefully uses `LocalRateLimiter` (process-local LRU token bucket).

Concrete settings for this release:

- In single-node deployments without Redis, run one Uvicorn worker per container so local limits are not split across processes.
- In multi-worker or multi-replica deployments (such as `docker-compose.yml`), configure `REDIS_URL` (e.g. `REDIS_URL=redis://redis:6379/0`) to enable shared distributed rate limiting across all instances.

## Application Rollback

Application rollback is handled at the deployment layer with immutable
container image tags. Policy rollback is separate and remains covered by
`core/policy_versioning.py`.

Required deployment setup:

- Build every release image with an immutable tag, such as a git SHA.
- Keep at least the previous known-good image available in the registry.
- Deploy by updating the service to the new immutable image tag, not by
  reusing `latest`.
- Record the deployed image tag in the release notes.

Rollback procedure:

1. Identify the last known-good image tag from the previous release.
2. Update the deployment service to that previous image tag.
3. Wait for the health check to pass on the replacement task/container.
4. Run a smoke check against `/health` and `GET /api/v1/whoami`.
5. Confirm no new 5xx spike in the deployment logs or metrics.

For local Docker Compose recovery, set the API image tag back to the
previous known-good value in the deployment environment and recreate only
the API service. Do not delete named volumes during an application
rollback; they hold runtime state and policy snapshots.
