"""
Copyright 2024, Zep Software, Inc.

Semantic Cache for Graphiti - accelerates search with vector similarity matching.
"""

import logging
from datetime import datetime
from threading import Lock
from time import time
from typing import Callable

from graphiti_core.cache.cache_entry import CacheEntry, CacheStats
from graphiti_core.cache.vector_index import IndexType, VectorIndex, create_vector_index
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.search.search_config import SearchResults

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic Cache for Graphiti search results.

    This cache uses vector similarity to find and return cached results
    for semantically similar queries, avoiding expensive graph database lookups.

    Features:
    - Vector-based semantic similarity matching
    - TTL (Time-To-Live) based expiration
    - Event-driven invalidation based on entity updates
    - Cache statistics for performance monitoring
    - Thread-safe operations

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity score to consider a cache hit (0.0-1.0).
        Higher values = stricter matching. Recommended: 0.85-0.95
    ttl_seconds : float
        Time-to-live for cache entries in seconds. Default: 300 (5 minutes)
    max_entries : int
        Maximum number of entries in the cache. Default: 1000
    embedding_dim : int
        Dimension of embedding vectors. Default: EMBEDDING_DIM from embedder config

    Example
    -------
    >>> cache = SemanticCache(similarity_threshold=0.9, ttl_seconds=300)
    >>>
    >>> # Check cache before search
    >>> cached = await cache.get(query_embedding, group_ids)
    >>> if cached:
    >>>     return cached.result
    >>>
    >>> # After search, store result
    >>> await cache.put(query, query_embedding, result, related_entities, group_ids)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
        embedding_dim: int = EMBEDDING_DIM,
        index_type: IndexType = 'auto',
        hnsw_m: int = 32,
        hnsw_ef_search: int = 64,
        hnsw_ef_construction: int = 200,
        enabled: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.embedding_dim = embedding_dim
        self.enabled = enabled

        # Vector index for semantic search
        self._index: VectorIndex = create_vector_index(
            embedding_dim,
            index_type=index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_search=hnsw_ef_search,
            hnsw_ef_construction=hnsw_ef_construction,
        )

        # Cache storage: index_id -> CacheEntry
        self._entries: dict[int, CacheEntry] = {}

        # Statistics
        self._stats = CacheStats()

        # Thread safety
        self._lock = Lock()

        logger.info(
            f'SemanticCache initialized: threshold={similarity_threshold}, '
            f'ttl={ttl_seconds}s, max_entries={max_entries}, index={self._index.index_type}'
        )

    def get(
        self,
        query_embedding: list[float],
        group_ids: list[str] | None = None,
    ) -> CacheEntry | None:
        """
        Look up a cached result by semantic similarity.

        Parameters
        ----------
        query_embedding : list[float]
            The embedding vector of the query.
        group_ids : list[str] | None
            Group IDs to filter cache entries.

        Returns
        -------
        CacheEntry | None
            The cached entry if found and valid, None otherwise.
        """
        if not self.enabled:
            return None

        start = time()

        with self._lock:
            # Clean expired entries first
            self._cleanup_expired()

            # Search for similar queries
            candidates = self._index.search(query_embedding, k=5)

            for index_id, similarity in candidates:
                if similarity < self.similarity_threshold:
                    # Since results are sorted by similarity, we can break early
                    break

                entry = self._entries.get(index_id)
                if entry is None:
                    continue

                # Check if entry matches group_ids
                if not entry.matches_group_ids(group_ids):
                    continue

                # Check TTL
                if entry.is_expired(self.ttl_seconds):
                    self._remove_entry(index_id)
                    self._stats.invalidations_by_ttl += 1
                    continue

                # Cache hit!
                entry.touch()
                latency = (time() - start) * 1000
                self._stats.record_hit(latency)

                logger.debug(
                    f'Cache HIT: similarity={similarity:.4f}, '
                    f'query="{entry.query[:50]}...", latency={latency:.2f}ms'
                )
                return entry

        # Cache miss (latency will be recorded after full search completes)
        self._stats.cache_misses += 1
        self._stats.total_queries += 1
        logger.debug('Cache MISS: will execute full search')
        return None

    def put(
        self,
        query: str,
        query_embedding: list[float],
        result: SearchResults,
        related_entities: set[str] | None = None,
        group_ids: list[str] | None = None,
    ) -> None:
        """
        Store a search result in the cache.

        Parameters
        ----------
        query : str
            The original query string.
        query_embedding : list[float]
            The embedding vector of the query.
        result : SearchResults
            The search results to cache.
        related_entities : set[str] | None
            Set of entity names related to this query (for invalidation).
        group_ids : list[str] | None
            Group IDs this entry belongs to.
        """
        if not self.enabled:
            return

        with self._lock:
            # Evict if at capacity
            if len(self._entries) >= self.max_entries:
                self._evict_lru()

            # Extract related entities from results if not provided
            if related_entities is None:
                related_entities = self._extract_entities_from_results(result)

            # Create cache entry
            entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                result=result,
                related_entities=related_entities,
                group_ids=group_ids,
            )

            # Add to vector index
            index_id = self._index.add(query_embedding)
            entry._index_id = index_id

            # Store entry
            self._entries[index_id] = entry

            logger.debug(
                f'Cache PUT: query="{query[:50]}...", '
                f'entities={len(related_entities)}, index_id={index_id}'
            )

    def invalidate_by_entities(self, entity_names: set[str]) -> int:
        """
        Invalidate all cache entries related to the given entities.

        This implements event-driven cache invalidation - when entities
        are updated in the graph, their cached results become stale.

        Parameters
        ----------
        entity_names : set[str]
            Set of entity names that were updated.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        if not self.enabled or not entity_names:
            return 0

        with self._lock:
            invalidated_ids = []

            for index_id, entry in self._entries.items():
                if entry.has_related_entity(entity_names):
                    invalidated_ids.append(index_id)

            for index_id in invalidated_ids:
                self._remove_entry(index_id)
                self._stats.invalidations_by_entity += 1

            if invalidated_ids:
                logger.info(
                    f'Cache invalidated {len(invalidated_ids)} entries '
                    f'for entities: {entity_names}'
                )

            return len(invalidated_ids)

    def invalidate_by_group_ids(self, group_ids: list[str]) -> int:
        """
        Invalidate all cache entries for the given group IDs.

        Parameters
        ----------
        group_ids : list[str]
            Group IDs to invalidate.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        if not self.enabled or not group_ids:
            return 0

        with self._lock:
            invalidated_ids = []
            group_id_set = set(group_ids)

            for index_id, entry in self._entries.items():
                if entry.group_ids and set(entry.group_ids) & group_id_set:
                    invalidated_ids.append(index_id)

            for index_id in invalidated_ids:
                self._remove_entry(index_id)

            return len(invalidated_ids)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._index.clear()
            self._entries.clear()
            logger.info('Cache cleared')

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def update_miss_latency(self, latency_ms: float) -> None:
        """
        Update the miss latency after full search completes.

        This should be called after the actual graph search is done,
        to reflect the true search latency (not just cache lookup latency).
        """
        self._stats.update_miss_latency(latency_ms)

    def _remove_entry(self, index_id: int) -> None:
        """Remove a single entry from cache (internal, caller holds lock)."""
        self._index.remove(index_id)
        self._entries.pop(index_id, None)

    def _cleanup_expired(self) -> None:
        """Remove expired entries (internal, caller holds lock)."""
        expired_ids = [
            index_id
            for index_id, entry in self._entries.items()
            if entry.is_expired(self.ttl_seconds)
        ]

        for index_id in expired_ids:
            self._remove_entry(index_id)
            self._stats.invalidations_by_ttl += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry (internal, caller holds lock)."""
        if not self._entries:
            return

        # Find LRU entry
        lru_id = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].last_accessed_at,
        )
        self._remove_entry(lru_id)
        logger.debug(f'Cache LRU eviction: index_id={lru_id}')

    def _extract_entities_from_results(self, result: SearchResults) -> set[str]:
        """Extract entity names from search results for invalidation tracking."""
        entities: set[str] = set()

        # Extract from nodes
        for node in result.nodes:
            entities.add(node.name)

        # Extract from edges (source and target node names if available)
        for edge in result.edges:
            # Edge has fact which often contains entity names
            # We could parse it, but it's safer to track by node
            pass

        return entities

    @property
    def size(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._entries)

    @property
    def index_type(self) -> str:
        """Return the active index type."""
        return self._index.index_type

    def __repr__(self) -> str:
        return (
            f'SemanticCache(size={self.size}, '
            f'threshold={self.similarity_threshold}, '
            f'hit_rate={self._stats.hit_rate:.2%})'
        )


# Global cache instance (singleton pattern)
_global_cache: SemanticCache | None = None


def get_semantic_cache(
    similarity_threshold: float = 0.9,
    ttl_seconds: float = 300.0,
    max_entries: int = 1000,
    index_type: IndexType = 'auto',
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
    hnsw_ef_construction: int = 200,
    enabled: bool = True,
) -> SemanticCache:
    """
    Get or create the global semantic cache instance.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity for cache hit.
    ttl_seconds : float
        Time-to-live for cache entries.
    max_entries : int
        Maximum cache size.
    enabled : bool
        Whether caching is enabled.

    Returns
    -------
    SemanticCache
        The global cache instance.
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = SemanticCache(
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            index_type=index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_search=hnsw_ef_search,
            hnsw_ef_construction=hnsw_ef_construction,
            enabled=enabled,
        )

    return _global_cache


def reset_semantic_cache() -> None:
    """Reset the global cache instance (for testing)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None
