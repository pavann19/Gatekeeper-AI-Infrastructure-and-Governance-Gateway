from core.config import EMBEDDING_MODEL, settings

# Lazy-loaded model — loaded only on first use, not at import time.
# This prevents CI failures when the package is mocked or unavailable.
DEFAULT_EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
_model = None

def _get_model():
    """Returns a cached SentenceTransformer instance (lazy initialization)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        revision = settings.EMBEDDING_MODEL_REVISION
        if revision is None and EMBEDDING_MODEL in ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"):
            revision = DEFAULT_EMBEDDING_REVISION
        _model = SentenceTransformer(EMBEDDING_MODEL, revision=revision)
    return _model

def get_embedding(text: str):
    """Generates a vector for the text."""
    return _get_model().encode(text, convert_to_tensor=True)

def cosine_similarity(vec1, vec2) -> float:
    """Calculates similarity."""
    from sentence_transformers import util  # noqa: PLC0415
    import torch

    if not isinstance(vec1, torch.Tensor):
        vec1 = torch.tensor(vec1)
    if not isinstance(vec2, torch.Tensor):
        vec2 = torch.tensor(vec2)

    return float(util.pytorch_cos_sim(vec1, vec2).item())
