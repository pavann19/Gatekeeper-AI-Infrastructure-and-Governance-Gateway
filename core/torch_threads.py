"""Configure Torch thread limits before model imports.

Assessment and detector pools already provide outer concurrency. Explicit
intra/inter-op limits reduce nested CPU oversubscription. This module is
imported before Torch-backed models; settings remain deployment-specific."""
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
