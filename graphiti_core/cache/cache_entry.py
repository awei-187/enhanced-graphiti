"""
Copyright 2024, Zep Software, Inc.

Semantic Cache Entry for Graphiti.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.datetime_utils import utc_now


@dataclass
class CacheEntry:
    """
    A single cache entry storing query embedding and search results.

    Attributes
    ----------
    query : str
        The original query string.
    query_embedding : list[float]
        The embedding vector of the query (for similarity matching).
    result : SearchResults
        The cached search results.
    related_entities : set[str]
        Set of entity names related to this query (for event-driven invalidation).
    group_ids : list[str] | None
        The group IDs this cache entry belongs to (for cache isolation).
    created_at : datetime
        When this cache entry was created (for TTL).
    last_accessed_at : datetime
        When this cache entry was last accessed (for LRU eviction).
    hit_count : int
        Number of times this cache entry was hit.
    """

    query: str
    query_embedding: list[float]
    result: SearchResults
    related_entities: set[str] = field(default_factory=set)
    group_ids: list[str] | None = None
    created_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime = field(default_factory=utc_now)
    hit_count: int = 0

    # Internal ID for vector index reference
    _index_id: int = -1

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this cache entry has expired based on TTL."""
        age = (utc_now() - self.created_at).total_seconds()
        return age > ttl_seconds

    def touch(self) -> None:
        """Update last accessed time and increment hit count."""
        self.last_accessed_at = utc_now()
        self.hit_count += 1

    def has_related_entity(self, entity_names: set[str]) -> bool:
        """Check if this cache entry is related to any of the given entities."""
        return bool(self.related_entities & entity_names)

    def matches_group_ids(self, group_ids: list[str] | None) -> bool:
        """Check if this cache entry matches the given group IDs."""
        if self.group_ids is None and group_ids is None:
            return True
        if self.group_ids is None or group_ids is None:
            return False
        return set(self.group_ids) == set(group_ids)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    invalidations_by_ttl: int = 0
    invalidations_by_entity: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit."""
        self.total_queries += 1
        self.cache_hits += 1
        # Running average
        self.avg_hit_latency_ms = (
            (self.avg_hit_latency_ms * (self.cache_hits - 1) + latency_ms)
            / self.cache_hits
        )

    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss."""
        self.total_queries += 1
        self.cache_misses += 1
        # Running average
        self.avg_miss_latency_ms = (
            (self.avg_miss_latency_ms * (self.cache_misses - 1) + latency_ms)
            / self.cache_misses
        )

    def update_miss_latency(self, latency_ms: float) -> None:
        """
        Update the miss latency after full search completes.

        This should be called after the actual search is done,
        not just after cache lookup, to reflect the true search latency.
        """
        if self.cache_misses > 0:
            # Update running average with the actual search latency
            # Since miss count was already incremented in get(), we use current count
            self.avg_miss_latency_ms = (
                (self.avg_miss_latency_ms * (self.cache_misses - 1) + latency_ms)
                / self.cache_misses
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging/export."""
        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f'{self.hit_rate:.2%}',
            'avg_hit_latency_ms': f'{self.avg_hit_latency_ms:.2f}',
            'avg_miss_latency_ms': f'{self.avg_miss_latency_ms:.2f}',
            'invalidations_by_ttl': self.invalidations_by_ttl,
            'invalidations_by_entity': self.invalidations_by_entity,
        }
