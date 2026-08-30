import numpy as np
from core.embeddings import get_embedding
from core.logger import get_logger

logger = get_logger(__name__)

class ScalableVectorStore:
    """
    FAISS-based vector store to replace O(N) list iteration.
    Pre-computes and indexes embeddings for O(log N) retrieval.
    FAISS is lazy-loaded on first use to allow CI mocking without the package.
    """
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.texts = []
        self._index = None  # Lazy-loaded

    def _get_index(self):
        """Lazy-initializes and returns the FAISS index."""
        if self._index is None:
            import faiss  # noqa: PLC0415
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add_texts(self, texts):
        if not texts:
            return

        import faiss  # noqa: PLC0415
        logger.info(f"Indexing {len(texts)} texts into FAISS...")
        vectors = []
        for t in texts:
            vec = get_embedding(t).cpu().numpy()
            vectors.append(vec)

        vec_matrix = np.vstack(vectors).astype('float32')
        faiss.normalize_L2(vec_matrix)
        self._get_index().add(vec_matrix)
        self.texts.extend(texts)

    def get_max_similarity(self, query_vec) -> float:
        """Returns the highest cosine similarity score for the query."""
        import faiss  # noqa: PLC0415
        index = self._get_index()
        if index.ntotal == 0:
            return 0.0

        if hasattr(query_vec, "cpu"):
            vec = query_vec.cpu().numpy()
        elif hasattr(query_vec, "numpy"):
            vec = query_vec.numpy()
        else:
            vec = np.array(query_vec, dtype="float32")

        vec = vec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(vec)

        scores, _ = index.search(vec, 1)
        return float(scores[0][0])

    def get_centroid(self):
        """Computes the semantic centroid of the store's vectors."""
        if not self.texts:
            return None
            
        vectors = []
        for t in self.texts:
            vectors.append(get_embedding(t).cpu().numpy())
            
        if not vectors:
            return None
            
        centroid = np.mean(vectors, axis=0)
        # Normalize to L2 so we can compute inner product easily
        import faiss  # noqa: PLC0415
        vec_matrix = centroid.reshape(1, -1).astype('float32')
        faiss.normalize_L2(vec_matrix)
        return vec_matrix[0]


# Global FAISS Stores (lazy-initialized on first use)
threat_store = ScalableVectorStore()
educational_store = ScalableVectorStore()
