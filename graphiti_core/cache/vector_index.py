"""
Copyright 2024, Zep Software, Inc.

Vector Index for Semantic Cache - supports FAISS and NumPy fallback.
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS, fall back to NumPy if not available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info('FAISS not available, using NumPy fallback for vector search')


class VectorIndex(ABC):
    """Abstract base class for vector index implementations."""

    @property
    @abstractmethod
    def index_type(self) -> str:
        """Return the index type identifier."""
        pass

    @abstractmethod
    def add(self, embedding: list[float]) -> int:
        """Add a vector to the index, return its ID."""
        pass

    @abstractmethod
    def search(self, query_vector: list[float], k: int = 1) -> list[tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Returns
        -------
        list[tuple[int, float]]
            List of (index_id, similarity_score) pairs, sorted by similarity descending.
        """
        pass

    @abstractmethod
    def remove(self, index_id: int) -> None:
        """Remove a vector from the index by its ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the index."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors in the index."""
        pass


class FAISSIndex(VectorIndex):
    """FAISS-based vector index using cosine similarity."""

    @property
    def index_type(self) -> str:
        return 'faiss_exact'

    def __init__(self, dimension: int):
        """
        Initialize FAISS index.

        Parameters
        ----------
        dimension : int
            The dimension of the embedding vectors.
        """
        self.dimension = dimension
        # Use Inner Product index (for cosine similarity, vectors must be normalized)
        self.index = faiss.IndexFlatIP(dimension)
        # Map internal FAISS IDs to our cache entry IDs
        self._id_counter = 0
        self._id_map: dict[int, int] = {}  # our_id -> faiss_position
        self._reverse_map: dict[int, int] = {}  # faiss_position -> our_id
        self._vectors: dict[int, np.ndarray] = {}  # our_id -> vector (for rebuild)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def add(self, embedding: list[float]) -> int:
        """Add a normalized vector to the index."""
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        normalized = self._normalize(vector.flatten()).reshape(1, -1)

        current_id = self._id_counter
        faiss_pos = self.index.ntotal

        self.index.add(normalized)

        self._id_map[current_id] = faiss_pos
        self._reverse_map[faiss_pos] = current_id
        self._vectors[current_id] = normalized.flatten()

        self._id_counter += 1
        return current_id

    def search(self, query_vector: list[float], k: int = 1) -> list[tuple[int, float]]:
        """Search for k nearest neighbors using cosine similarity."""
        if self.index.ntotal == 0:
            return []

        vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        normalized = self._normalize(vector.flatten()).reshape(1, -1)

        # Search for k results (or all if less than k)
        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(normalized, actual_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx in self._reverse_map:
                our_id = self._reverse_map[idx]
                # Convert inner product to cosine similarity (already in [0,1] for normalized vectors)
                results.append((our_id, float(score)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def remove(self, index_id: int) -> None:
        """Remove a vector by rebuilding the index without it."""
        if index_id not in self._id_map:
            return

        del self._vectors[index_id]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from remaining vectors."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self._id_map.clear()
        self._reverse_map.clear()

        for our_id, vector in self._vectors.items():
            faiss_pos = self.index.ntotal
            self.index.add(vector.reshape(1, -1))
            self._id_map[our_id] = faiss_pos
            self._reverse_map[faiss_pos] = our_id

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self._id_map.clear()
        self._reverse_map.clear()
        self._vectors.clear()
        self._id_counter = 0

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal


class NumpyIndex(VectorIndex):
    """NumPy-based vector index fallback (for small-scale use without FAISS)."""

    @property
    def index_type(self) -> str:
        return 'numpy'

    def __init__(self, dimension: int):
        """
        Initialize NumPy index.

        Parameters
        ----------
        dimension : int
            The dimension of the embedding vectors.
        """
        self.dimension = dimension
        self._id_counter = 0
        self._vectors: dict[int, np.ndarray] = {}

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        v1_norm = self._normalize(v1)
        v2_norm = self._normalize(v2)
        return float(np.dot(v1_norm, v2_norm))

    def add(self, embedding: list[float]) -> int:
        """Add a vector to the index."""
        vector = np.array(embedding, dtype=np.float32)
        current_id = self._id_counter
        self._vectors[current_id] = vector
        self._id_counter += 1
        return current_id

    def search(self, query_vector: list[float], k: int = 1) -> list[tuple[int, float]]:
        """Search for k nearest neighbors using cosine similarity."""
        if not self._vectors:
            return []

        query = np.array(query_vector, dtype=np.float32)
        similarities = []

        for idx, vector in self._vectors.items():
            sim = self._cosine_similarity(query, vector)
            similarities.append((idx, sim))

        # Sort by similarity descending and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def remove(self, index_id: int) -> None:
        """Remove a vector from the index by its ID."""
        self._vectors.pop(index_id, None)

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._vectors.clear()
        self._id_counter = 0

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._vectors)


class FAISSHNSWIndex(VectorIndex):
    """FAISS HNSW index for approximate cosine similarity search."""

    def __init__(self, dimension: int, m: int = 32, ef_search: int = 64, ef_construction: int = 200):
        """
        Initialize FAISS HNSW index.

        Parameters
        ----------
        dimension : int
            The dimension of the embedding vectors.
        m : int
            HNSW graph degree (higher improves recall, increases memory/time).
        ef_search : int
            Search depth parameter controlling recall/latency tradeoff.
        ef_construction : int
            Construction depth parameter controlling index build quality.
        """
        self.dimension = dimension
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction

        self.index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.hnsw.efConstruction = ef_construction

        self._id_counter = 0
        self._id_map: dict[int, int] = {}
        self._reverse_map: dict[int, int] = {}
        self._vectors: dict[int, np.ndarray] = {}

    @property
    def index_type(self) -> str:
        return 'faiss_hnsw'

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def add(self, embedding: list[float]) -> int:
        """Add a normalized vector to the index."""
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        normalized = self._normalize(vector.flatten()).reshape(1, -1)

        current_id = self._id_counter
        faiss_pos = self.index.ntotal

        self.index.add(normalized)

        self._id_map[current_id] = faiss_pos
        self._reverse_map[faiss_pos] = current_id
        self._vectors[current_id] = normalized.flatten()

        self._id_counter += 1
        return current_id

    def search(self, query_vector: list[float], k: int = 1) -> list[tuple[int, float]]:
        """Search for k nearest neighbors using cosine similarity."""
        if self.index.ntotal == 0:
            return []

        vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        normalized = self._normalize(vector.flatten()).reshape(1, -1)

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(normalized, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self._reverse_map:
                our_id = self._reverse_map[idx]
                results.append((our_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def remove(self, index_id: int) -> None:
        """Remove a vector by rebuilding the index without it."""
        if index_id not in self._id_map:
            return

        del self._vectors[index_id]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from remaining vectors."""
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = self.ef_search
        self.index.hnsw.efConstruction = self.ef_construction
        self._id_map.clear()
        self._reverse_map.clear()

        for our_id, vector in self._vectors.items():
            faiss_pos = self.index.ntotal
            self.index.add(vector.reshape(1, -1))
            self._id_map[our_id] = faiss_pos
            self._reverse_map[faiss_pos] = our_id

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = self.ef_search
        self.index.hnsw.efConstruction = self.ef_construction
        self._id_map.clear()
        self._reverse_map.clear()
        self._vectors.clear()
        self._id_counter = 0

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal


IndexType = Literal['auto', 'faiss_exact', 'faiss_hnsw', 'numpy']


def create_vector_index(
    dimension: int,
    index_type: IndexType = 'auto',
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
    hnsw_ef_construction: int = 200,
) -> VectorIndex:
    """
    Factory function to create the best available vector index.

    Parameters
    ----------
    dimension : int
        The dimension of the embedding vectors.

    Returns
    -------
    VectorIndex
        FAISS index if available, otherwise NumPy fallback.
    """
    if index_type == 'numpy':
        logger.info(f'Creating NumPy index with dimension {dimension}')
        return NumpyIndex(dimension)

    if not FAISS_AVAILABLE:
        logger.info('FAISS not available, using NumPy fallback for vector search')
        return NumpyIndex(dimension)

    if index_type in ('auto', 'faiss_exact'):
        logger.info(f'Creating FAISS exact index with dimension {dimension}')
        return FAISSIndex(dimension)

    if index_type == 'faiss_hnsw':
        logger.info(
            'Creating FAISS HNSW index with dimension %d (m=%d, ef_search=%d, ef_construction=%d)',
            dimension,
            hnsw_m,
            hnsw_ef_search,
            hnsw_ef_construction,
        )
        return FAISSHNSWIndex(
            dimension,
            m=hnsw_m,
            ef_search=hnsw_ef_search,
            ef_construction=hnsw_ef_construction,
        )

    logger.info(f'Unknown index_type={index_type}, falling back to NumPy index')
    return NumpyIndex(dimension)
