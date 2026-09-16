"""
Pins torch's thread pools before any model loads. Must be imported before
the first `import torch` anywhere in the process (transitively, via
transformers/sentence-transformers) — torch's thread pools initialize on
first use and cannot be resized afterward.

See core/config.py's TORCH_INTRA_OP_THREADS/TORCH_INTER_OP_THREADS
docstring and docs/perf/01-baseline-analysis.md for why this exists:
8 detectors dispatched in parallel per assessment x up to
ASSESS_MAX_CONCURRENCY concurrent assessments x torch's own unpinned
default (6 intra-op threads on this box) oversubscribes a 12-core machine
by design, not by accident.
"""
from __future__ import annotations

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

_configured = False


def configure_torch_threads() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    try:
        import torch
        torch.set_num_threads(settings.TORCH_INTRA_OP_THREADS)
        torch.set_num_interop_threads(settings.TORCH_INTER_OP_THREADS)
        logger.info(
            "torch threads pinned: intra_op=%d inter_op=%d (was unpinned before)",
            settings.TORCH_INTRA_OP_THREADS, settings.TORCH_INTER_OP_THREADS,
        )
    except RuntimeError as e:
        # torch.set_num_interop_threads raises if interop threads were
        # already used (e.g. a prior torch call happened before this ran) --
        # fail soft and loud rather than crash the process over a thread-
        # count tuning knob.
        logger.warning("could not pin torch threads (already initialized?): %s", e)
    except ImportError:
        pass  # torch not installed in this environment (e.g. CI unit-test tier)


configure_torch_threads()
