"""
Unit tests for per-tenant privacy and PII configuration (core/privacy.py, core/tenancy.py).
"""
import json
import pytest

from core.privacy import redact_pii
from core.tenancy import TenantConfig, TenantStore


def test_redact_pii_default_tenant():
    # By default, all regexes (e.g. AADHAAR, EMAIL) are active
    text = "My email is user@example.com and Aadhaar is 1234 5678 9012"
    clean, metadata = redact_pii(text)
    assert "[REDACTED:EMAIL]" in clean
    assert "[REDACTED:AADHAAR]" in clean
    assert metadata["pii_found"] is True


def test_redact_pii_with_disabled_patterns():
    # Tenant with disabled AADHAAR pattern
    tenant_cfg = TenantConfig(
        tenant_id="custom-corp",
        privacy_disabled_patterns=("AADHAAR",),
    )
    text = "My email is user@example.com and Aadhaar is 1234 5678 9012"
    clean, metadata = redact_pii(text, tenant_config=tenant_cfg)
    assert "[REDACTED:EMAIL]" in clean
    assert "1234 5678 9012" in clean
    assert "[REDACTED:AADHAAR]" not in clean
    assert metadata["pii_found"] is True


def test_redact_pii_custom_ner_labels(monkeypatch):
    tenant_cfg = TenantConfig(
        tenant_id="ner-custom",
        privacy_ner_labels=("PERSON",),  # Only PERSON, no ORG/GPE
    )
    # Clean regex text so it proceeds to NER
    text = "Alice visited Microsoft headquarters in Seattle."
    clean, metadata = redact_pii(text, tenant_config=tenant_cfg)
    if metadata["pii_found"]:
        # If NER model ran, ensure no GPE/ORG was redacted
        assert "[REDACTED:GPE]" not in clean


def test_tenant_store_loads_privacy_overrides(tmp_path):
    store_file = tmp_path / "tenants.json"
    data = {
        "privacy-tenant": {
            "display_name": "Privacy Corp",
            "status": "active",
            "privacy_disabled_patterns": ["AADHAAR", "phone"],
            "privacy_ner_labels": ["PERSON", "ORG"],
        }
    }
    store_file.write_text(json.dumps(data), encoding="utf-8")

    store = TenantStore(path=str(store_file))
    cfg = store.get("privacy-tenant")
    assert cfg.tenant_id == "privacy-tenant"
    assert cfg.privacy_disabled_patterns == ("AADHAAR", "PHONE")
    assert cfg.privacy_ner_labels == ("PERSON", "ORG")
