# Deployment Runbook

This runbook records the release-layer decisions that are outside the
application code but must be true for an application release.

## Runtime Shape

The approved release shape is a single API worker process per deployed
instance.

`core/rate_limit.py` and `core/token_quota.py` keep their counters in
process memory. That is correct for one worker, but a multi-worker or
multi-replica deployment would multiply the effective limit by the number
of workers. Do not deploy this release with multiple API workers unless
the rate limiter and token quota are first moved to a shared store such as
Redis.

Concrete settings for this release:

- Run one Uvicorn/Gunicorn worker for the API container.
- Scale vertically before scaling the API process count.
- Treat any change to more than one API worker or replica as a release
  architecture change requiring Redis-backed distributed rate limiting and
  quota accounting.

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
