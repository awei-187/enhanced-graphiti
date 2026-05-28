"""
Copyright 2024, Zep Software, Inc.

Semantic Cache module for Graphiti.
"""

from graphiti_core.cache.cache_entry import CacheEntry, CacheStats
from graphiti_core.cache.semantic_cache import (
    SemanticCache,
    get_semantic_cache,
    reset_semantic_cache,
)
from graphiti_core.cache.subgraph_cache import (
    SubgraphCache,
    SubgraphCacheEntry,
    SubgraphCacheStats,
    SubgraphSearchResult,
)
from graphiti_core.cache.vector_index import (
    FAISS_AVAILABLE,
    VectorIndex,
    create_vector_index,
)

__all__ = [
    'CacheEntry',
    'CacheStats',
    'SemanticCache',
    'SubgraphCache',
    'SubgraphCacheEntry',
    'SubgraphCacheStats',
    'SubgraphSearchResult',
    'get_semantic_cache',
    'reset_semantic_cache',
    'VectorIndex',
    'create_vector_index',
    'FAISS_AVAILABLE',
]
