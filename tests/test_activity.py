"""
Tests for core/activity.py's reverse-chunk tail read over the audit log.
Uses real files on disk (tmp_path), including ones bigger than
core.activity.CHUNK_SIZE, so these tests exercise the actual multi-chunk
reconstruction logic, not just the single-chunk happy path.
"""
import json

import core.activity as activity_mod
from core.activity import find_by_request_id, get_recent_activity


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _event(event_type="input_assessment", tenant="acme", capability="GENERAL", **extra):
    record = {
        "timestamp": "2026-08-24T00:00:00",
        "event_type": event_type,
        "tenant": tenant,
        "capability": capability,
        "decision": "ALLOW",
    }
    record.update(extra)
    return record


def test_missing_file_returns_empty_not_an_error(tmp_path):
    result = get_recent_activity(path=str(tmp_path / "does_not_exist.jsonl"))
    assert result == {"events": [], "scan_truncated": False}


def test_empty_file_returns_empty(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text("", encoding="utf-8")
    result = get_recent_activity(path=str(p))
    assert result["events"] == []


def test_events_come_back_newest_first(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_event(request_id=str(i)) for i in range(5)])
    result = get_recent_activity(path=str(p))
    ids = [e["request_id"] for e in result["events"]]
    assert ids == ["4", "3", "2", "1", "0"]


def test_limit_caps_results(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_event(request_id=str(i)) for i in range(20)])
    result = get_recent_activity(limit=3, path=str(p))
    assert len(result["events"]) == 3
    assert [e["request_id"] for e in result["events"]] == ["19", "18", "17"]


def test_limit_above_max_is_clamped(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_event(request_id=str(i)) for i in range(300)])
    result = get_recent_activity(limit=10_000, path=str(p))
    assert len(result["events"]) == activity_mod.MAX_LIMIT


def test_tenant_filter(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [
        _event(tenant="acme", request_id="a1"),
        _event(tenant="beta", request_id="b1"),
        _event(tenant="acme", request_id="a2"),
    ])
    result = get_recent_activity(tenant="acme", path=str(p))
    assert {e["request_id"] for e in result["events"]} == {"a1", "a2"}


def test_event_type_filter(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [
        _event(event_type="tool_call", request_id="t1"),
        _event(event_type="gateway_call", request_id="g1"),
        _event(event_type="tool_call", request_id="t2"),
    ])
    result = get_recent_activity(event_types=["tool_call"], path=str(p))
    assert {e["request_id"] for e in result["events"]} == {"t1", "t2"}


def test_malformed_line_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "audit.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(_event(request_id="good1")) + "\n")
        f.write("{ not valid json\n")
        f.write(json.dumps(_event(request_id="good2")) + "\n")
    result = get_recent_activity(path=str(p))
    ids = [e["request_id"] for e in result["events"]]
    assert ids == ["good2", "good1"]


def test_missing_event_type_normalises_to_legacy(tmp_path):
    p = tmp_path / "audit.jsonl"
    legacy_record = {"timestamp": "2025-01-01T00:00:00", "decision": "BLOCK",
                     "risk": "HIGH", "role": "student"}
    _write_jsonl(p, [legacy_record])
    result = get_recent_activity(path=str(p))
    assert result["events"][0]["event_type"] == "legacy"
    assert result["events"][0]["tenant"] == "unset"
    assert result["events"][0]["capability"] == "unset"


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "audit.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(_event(request_id="a")) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps(_event(request_id="b")) + "\n")
    result = get_recent_activity(path=str(p))
    assert [e["request_id"] for e in result["events"]] == ["b", "a"]


def test_reconstructs_lines_spanning_multiple_chunks(tmp_path, monkeypatch):
    """Forces a tiny chunk size so a single JSON line is guaranteed to
    straddle more than one read -- the real multi-chunk carry-over path,
    not just the single-read happy case every other test exercises."""
    monkeypatch.setattr(activity_mod, "CHUNK_SIZE", 16)
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_event(request_id=str(i), tenant="tenant-with-a-longer-name")
                     for i in range(30)])
    result = get_recent_activity(limit=10, path=str(p))
    assert len(result["events"]) == 10
    assert [e["request_id"] for e in result["events"]] == [str(i) for i in range(29, 19, -1)]
    for e in result["events"]:
        assert e["tenant"] == "tenant-with-a-longer-name"


def test_scan_truncated_flag_set_when_byte_cap_hit_before_enough_matches(tmp_path, monkeypatch):
    """A filter matching almost nothing must not silently return an
    incomplete result set that looks like 'there just weren't more events'
    -- scan_truncated tells the caller the scan gave up, not that the data
    ran out."""
    monkeypatch.setattr(activity_mod, "MAX_BYTES_SCANNED", 200)
    p = tmp_path / "audit.jsonl"
    # All 'beta' tenant except one far-back 'acme' record the tiny byte cap
    # will never reach.
    records = [_event(tenant="acme", request_id="only-match")]
    records += [_event(tenant="beta", request_id=str(i)) for i in range(200)]
    _write_jsonl(p, records)
    result = get_recent_activity(tenant="acme", limit=5, path=str(p))
    assert result["events"] == []
    assert result["scan_truncated"] is True


def test_real_default_path_used_when_none_given(monkeypatch, tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_event(request_id="only")])
    monkeypatch.setattr("core.activity.settings.AUDIT_LOG_PATH", str(p))
    result = get_recent_activity()
    assert result["events"][0]["request_id"] == "only"


# --- find_by_request_id -------------------------------------------------------

def test_trace_returns_only_matching_request_id_in_chronological_order(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [
        _event(request_id="target", event_type="input_assessment"),
        _event(request_id="other", event_type="input_assessment"),
        _event(request_id="target", event_type="gateway_call"),
        _event(request_id="other", event_type="tool_call"),
        _event(request_id="target", event_type="output_assessment"),
    ])
    result = find_by_request_id("target", path=str(p))
    types = [e["event_type"] for e in result["events"]]
    assert types == ["input_assessment", "gateway_call", "output_assessment"]


def test_trace_no_matches_is_an_empty_list_not_an_error(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_event(request_id="unrelated")])
    result = find_by_request_id("does-not-exist", path=str(p))
    assert result == {"events": [], "scan_truncated": False}


def test_trace_tenant_filter(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [
        _event(request_id="shared-id", tenant="acme"),
        _event(request_id="shared-id", tenant="beta"),
    ])
    result = find_by_request_id("shared-id", tenant="acme", path=str(p))
    assert len(result["events"]) == 1
    assert result["events"][0]["tenant"] == "acme"


def test_trace_missing_file_is_empty(tmp_path):
    result = find_by_request_id("anything", path=str(tmp_path / "nope.jsonl"))
    assert result == {"events": [], "scan_truncated": False}


def test_trace_scan_truncated_when_byte_cap_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_mod, "MAX_BYTES_SCANNED", 200)
    p = tmp_path / "audit.jsonl"
    records = [_event(request_id="target")]
    records += [_event(request_id=str(i), tenant="beta") for i in range(200)]
    _write_jsonl(p, records)
    result = find_by_request_id("target", path=str(p))
    assert result["events"] == []
    assert result["scan_truncated"] is True


def test_trace_spans_multiple_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_mod, "CHUNK_SIZE", 16)
    p = tmp_path / "audit.jsonl"
    records = [_event(request_id="target", event_type="input_assessment")]
    records += [_event(request_id=str(i)) for i in range(30)]
    records.append(_event(request_id="target", event_type="tool_call"))
    _write_jsonl(p, records)
    result = find_by_request_id("target", path=str(p))
    assert [e["event_type"] for e in result["events"]] == ["input_assessment", "tool_call"]
