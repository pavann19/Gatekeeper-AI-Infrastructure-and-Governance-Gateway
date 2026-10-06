"""Bounded cross-process locking for local persistent state.

Requires a local filesystem with working advisory locks. Network filesystem
and independently mounted replica volumes are outside this contract.
"""
import os
import sys
import threading
import time
from contextlib import contextmanager
from weakref import WeakValueDictionary

_registry_guard = threading.Lock()
_locks: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()


@contextmanager
def state_lock(path: str, timeout: float = 5.0):
    from core.request_budget import remaining
    timeout = remaining(timeout)
    key = os.path.normcase(os.path.abspath(path))
    with _registry_guard:
        lock = _locks.setdefault(key, threading.RLock())
    if not lock.acquire(timeout=timeout):
        raise TimeoutError("Persistent state is busy")
    handle = None
    acquired = False
    try:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        handle = open(key + '.lock', 'a+b')
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Persistent state is busy") from None
                time.sleep(0.01)
        yield
    finally:
        if handle is not None:
            if acquired:
                if sys.platform == 'win32':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        lock.release()
