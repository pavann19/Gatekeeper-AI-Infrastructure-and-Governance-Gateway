"""Shared admission checks for HTTP and MCP transports."""
from fastapi import HTTPException

from core.config import settings


def require_admission(principal, tenant_config, require_auth=False):
    if require_auth and not principal.authenticated:
        raise HTTPException(401, "Authentication required. Present a valid API key.",
                            headers={"WWW-Authenticate": "Bearer"})
    if tenant_config.suspended:
        raise HTTPException(403, "Tenant admission is unavailable or suspended")


def owns_tenant_record(principal, record):
    return principal.authenticated and (
        principal.capability == settings.CAPABILITY_INTERNAL or record.get("tenant") == principal.tenant)
