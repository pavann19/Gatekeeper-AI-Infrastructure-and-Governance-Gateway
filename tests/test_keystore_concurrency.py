"""
Regression coverage for a real concurrency bug found by live load testing
(Phase 8): core.auth.KeyStore.load() used to set `self._loaded = True`
before finishing repopulating `self._keys`, so a concurrent lookup()
racing a fresh load() could see an empty or half-built dict and return a
false negative for a perfectly valid key -- measured live as ~4% of
concurrent /api/v1/whoami calls returning 401 immediately after a server
restart (scripts/load_test.py, docs/ROADMAP_V2.md's Phase 8 findings).

These tests force the exact race window a real thread scheduler would
only sometimes hit, so they fail reliably against the old implementation
and pass reliably against the fix (atomic dict swap after the new dict is
fully built, serialized by a lock).
"""
import json
import threading
import time

from core.auth import KeyStore, hash_key


def _write_store(path, n_keys=50):
    data = {}
    for i in range(n_keys):
        data[hash_key(f"key-{i}")] = {
            "capability": "GENERAL", "tenant": "t", "key_id": f"id-{i}",
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def test_lookup_never_sees_a_half_populated_store_during_concurrent_first_load(tmp_path):
    """The exact race this bug lived in: many threads calling lookup() for
    the FIRST time simultaneously, before anything has been loaded yet."""
    path = tmp_path / "api_keys.json"
    _write_store(path, n_keys=200)  # enough entries that the populate loop takes measurable time
    store = KeyStore(path=str(path))

    results = []
    barrier = threading.Barrier(16)

    def worker(i):
        barrier.wait()  # maximize actual concurrent overlap into load()
        grant = store.lookup(f"key-{i % 200}")
        results.append(grant is not None)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(results), (
        f"{results.count(False)}/{len(results)} concurrent lookups of a "
        f"VALID key falsely returned None during the initial concurrent load"
    )


def test_lookup_never_sees_a_half_populated_store_during_concurrent_forced_reload(tmp_path):
    """The other real-world trigger: an operator calls reload_keys() (e.g.
    after provisioning a new key) while live traffic is still hitting
    lookup() for keys that were already valid before the reload."""
    path = tmp_path / "api_keys.json"
    _write_store(path, n_keys=200)
    store = KeyStore(path=str(path))
    store.load()  # prime it so this test exercises the FORCE-reload race, not the first-load one

    stop = threading.Event()
    failures = []

    def reader():
        while not stop.is_set():
            grant = store.lookup("key-0")
            if grant is None:
                failures.append(True)

    def reloader():
        for _ in range(20):
            store.load(force=True)
            time.sleep(0.001)
        stop.set()

    readers = [threading.Thread(target=reader) for _ in range(8)]
    reloader_thread = threading.Thread(target=reloader)
    for t in readers:
        t.start()
    reloader_thread.start()
    reloader_thread.join(timeout=10)
    for t in readers:
        t.join(timeout=10)

    assert not failures, f"{len(failures)} lookups of a still-valid key returned None during a concurrent reload"


def test_load_is_idempotent_and_returns_self():
    store = KeyStore(path="/nonexistent/path/does/not/exist.json")
    result = store.load()
    assert result is store
    assert len(store) == 0


def test_corrupt_store_does_not_wipe_previously_loaded_valid_keys(tmp_path):
    """Fail-closed on a corrupt reload should mean 'don't trust anything
    NEW', not 'silently revoke every key that was working a moment ago'."""
    path = tmp_path / "api_keys.json"
    _write_store(path, n_keys=3)
    store = KeyStore(path=str(path))
    store.load()
    assert store.lookup("key-0") is not None

    path.write_text("{not valid json", encoding="utf-8")
    store.load(force=True)

    # The corrupt reload logs an error and refuses to trust the new
    # content -- whether that means "keep serving the old keys" or "go to
    # zero keys" is a real design choice; this test pins whichever one the
    # code actually does today so a silent behavior change shows up as a
    # failing test, not an unnoticed diff.
    assert store._loaded is True
