from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.cache.subgraph_cache import SubgraphCache
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.search.search import search
from graphiti_core.search.search_config import (
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters


class DummyCrossEncoder:
    async def rank(self, _query: str, texts: list[str]) -> list[tuple[str, float]]:
        return [(text, 1.0) for text in texts]


class DummyEmbedder:
    async def create(self, input_data: list[str]) -> list[float]:
        return [0.0, 1.0]


class DummySemanticCache:
    def __init__(self, result: SearchResults) -> None:
        self.result = result
        self.get_calls = 0

    def get(self, _query_embedding: list[float], _group_ids: list[str] | None = None):
        self.get_calls += 1
        return SimpleNamespace(result=self.result, hit_count=1)

    def put(
        self,
        _query: str,
        _query_embedding: list[float],
        _result: SearchResults,
        _related_entities: set[str] | None = None,
        _group_ids: list[str] | None = None,
    ) -> None:
        return None

    def update_miss_latency(self, _latency_ms: float) -> None:
        return None


class DummyDriver:
    search_interface = None


def _node(name: str, uuid: str, embedding: list[float]) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=name,
        group_id='group-1',
        labels=['Entity'],
        created_at=datetime.now(timezone.utc),
        name_embedding=embedding,
    )


def _edge(
    uuid: str,
    source_node_uuid: str,
    target_node_uuid: str,
    fact: str,
    embedding: list[float],
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        name='RELATES_TO',
        group_id='group-1',
        source_node_uuid=source_node_uuid,
        target_node_uuid=target_node_uuid,
        created_at=datetime.now(timezone.utc),
        fact=fact,
        fact_embedding=embedding,
    )


def _community(name: str, uuid: str, embedding: list[float]) -> CommunityNode:
    return CommunityNode(
        uuid=uuid,
        name=name,
        group_id='group-1',
        created_at=datetime.now(timezone.utc),
        summary='',
        name_embedding=embedding,
    )


def _hybrid_config() -> SearchConfig:
    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf,
        ),
        limit=5,
    )


def _edge_config() -> SearchConfig:
    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        limit=5,
    )


@pytest.mark.asyncio
async def test_subgraph_cache_returns_confident_local_result() -> None:
    cache = SubgraphCache(min_top1_score=0.0)
    config = _hybrid_config()
    filters = SearchFilters()
    alice = _node('Alice', 'node-1', [1.0, 0.0])
    bob = _node('Bob', 'node-2', [0.0, 1.0])
    edge = _edge('edge-1', alice.uuid, bob.uuid, 'Alice knows Bob', [1.0, 0.0])

    cache.put(SearchResults(nodes=[alice, bob], edges=[edge]), ['group-1'], config, filters)

    result = await cache.get(
        query='Alice',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is not None
    assert [node.uuid for node in result.result.nodes] == ['node-1']
    assert [edge.uuid for edge in result.result.edges] == ['edge-1']
    assert cache.get_stats().cache_hits == 1


@pytest.mark.asyncio
async def test_subgraph_cache_returns_edge_only_local_result() -> None:
    cache = SubgraphCache(min_top1_score=0.0)
    config = _edge_config()
    filters = SearchFilters()
    alice = _node('Alice', 'node-1', [1.0, 0.0])
    edge = _edge('edge-1', alice.uuid, 'node-2', 'Alice knows Bob', [1.0, 0.0])

    cache.put(SearchResults(edges=[edge]), ['group-1'], config, filters)

    result = await cache.get(
        query='Alice',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is not None
    assert [edge.uuid for edge in result.result.edges] == ['edge-1']
    assert result.confidence_score <= result.top1_score
    assert cache.get_stats().cache_hits == 1


@pytest.mark.asyncio
async def test_subgraph_cache_uses_non_node_confidence_when_node_results_are_empty() -> None:
    cache = SubgraphCache(min_top1_score=0.5)
    config = _hybrid_config()
    filters = SearchFilters(node_labels=['Person'])
    alice = _node('Alice', 'node-1', [1.0, 0.0])
    edge = _edge('edge-1', alice.uuid, 'node-2', 'Alice knows Bob', [1.0, 0.0])

    cache.put(SearchResults(nodes=[alice], edges=[edge]), ['group-1'], config, filters)

    result = await cache.get(
        query='Alice',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is not None
    assert result.result.nodes == []
    assert [cache_edge.uuid for cache_edge in result.result.edges] == ['edge-1']
    assert result.confidence_score <= result.top1_score
    assert cache.get_stats().cache_hits == 1


@pytest.mark.asyncio
async def test_subgraph_cache_penalizes_singleton_confidence() -> None:
    cache = SubgraphCache(min_top1_score=0.0)
    config = SearchConfig(
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.cross_encoder,
        ),
        limit=5,
    )
    filters = SearchFilters()
    alice = _node('Alice', 'node-1', [1.0, 0.0])

    cache.put(SearchResults(nodes=[alice]), ['group-1'], config, filters)

    result = await cache.get(
        query='Alice',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is not None
    assert result.top2_score == 0.0
    assert result.top1_score == 1.0
    assert result.confidence_score < 1.0


@pytest.mark.asyncio
async def test_subgraph_cache_falls_back_when_node_score_below_threshold() -> None:
    cache = SubgraphCache(min_top1_score=0.5)
    config = _hybrid_config()
    filters = SearchFilters()
    alice = _node('Alice', 'node-1', [1.0, 0.0])
    edge = _edge('edge-1', alice.uuid, 'node-2', 'Alice knows Bob', [1.0, 0.0])

    cache.put(SearchResults(nodes=[alice], edges=[edge]), ['group-1'], config, filters)

    result = await cache.get(
        query='Carol',
        query_vector=[0.0, 1.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    stats = cache.get_stats()
    assert result is None
    assert stats.cache_misses == 1
    assert stats.fallbacks_by_low_score == 1


@pytest.mark.asyncio
async def test_subgraph_cache_selects_best_node_confidence() -> None:
    cache = SubgraphCache(min_top1_score=0.5)
    config = _hybrid_config()
    filters = SearchFilters()
    weak_node = _node('Alice', 'weak-node', [0.0, 1.0])
    strong_node = _node('Carol', 'strong-node', [1.0, 0.0])
    strong_edge = _edge(
        'strong-edge',
        weak_node.uuid,
        strong_node.uuid,
        'Alice repeated many searchable words',
        [1.0, 0.0],
    )

    cache.put(
        SearchResults(nodes=[weak_node], edges=[strong_edge]),
        ['group-1'],
        config,
        filters,
    )
    cache.put(SearchResults(nodes=[strong_node]), ['group-1'], config, filters)

    result = await cache.get(
        query='Carol',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is not None
    assert [node.uuid for node in result.result.nodes] == ['strong-node']
    assert cache.get_stats().cache_hits == 1


@pytest.mark.asyncio
async def test_subgraph_cache_uses_configured_community_methods() -> None:
    cache = SubgraphCache(min_top1_score=0.0)
    config = SearchConfig(
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.rrf,
        ),
        limit=5,
    )
    filters = SearchFilters()
    community = _community('Alice community', 'community-1', [0.0, 1.0])

    cache.put(SearchResults(communities=[community]), ['group-1'], config, filters)

    result = await cache.get(
        query='Alice',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is None
    assert cache.get_stats().cache_misses == 1


@pytest.mark.asyncio
async def test_search_low_score_subgraph_miss_skips_semantic_cache(monkeypatch) -> None:
    stale_node = _node('Cached Alice', 'semantic-node', [1.0, 0.0])
    fresh_node = _node('Database Carol', 'database-node', [0.0, 1.0])

    async def fake_node_search(*_args, **_kwargs):
        return [fresh_node], [1.0]

    monkeypatch.setattr('graphiti_core.search.search.node_search', fake_node_search)
    monkeypatch.setattr('graphiti_core.search.search.edge_search', _empty_search)
    monkeypatch.setattr('graphiti_core.search.search.episode_search', _empty_search)
    monkeypatch.setattr('graphiti_core.search.search.community_search', _empty_search)

    config = SearchConfig(
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf,
        ),
        limit=5,
    )
    filters = SearchFilters()
    subgraph_cache = SubgraphCache(min_top1_score=0.5)
    subgraph_cache.put(
        SearchResults(nodes=[_node('Alice', 'node-1', [1.0, 0.0])]),
        ['group-1'],
        config,
        filters,
    )
    semantic_cache = DummySemanticCache(SearchResults(nodes=[stale_node]))
    clients = SimpleNamespace(
        driver=DummyDriver(),
        llm_client=None,
        embedder=DummyEmbedder(),
        cross_encoder=DummyCrossEncoder(),
    )

    result = await search(
        clients=clients,
        query='Carol',
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cache=semantic_cache,
        use_cache=True,
        subgraph_cache=subgraph_cache,
        use_subgraph_cache=True,
    )

    assert [node.uuid for node in result.nodes] == ['database-node']
    assert semantic_cache.get_calls == 0
    assert subgraph_cache.get_stats().fallbacks_by_low_score == 1


@pytest.mark.asyncio
async def test_subgraph_cache_skips_structured_edge_search() -> None:
    cache = SubgraphCache(min_top1_score=0.0)
    config = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.structured],
            reranker=EdgeReranker.cross_encoder,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25],
            reranker=NodeReranker.rrf,
        ),
    )
    filters = SearchFilters()
    alice = _node('Alice', 'node-1', [1.0, 0.0])

    cache.put(SearchResults(nodes=[alice]), ['group-1'], config, filters)

    result = await cache.get(
        query='Alice',
        query_vector=[1.0, 0.0],
        group_ids=['group-1'],
        config=config,
        search_filter=filters,
        cross_encoder=DummyCrossEncoder(),
    )

    assert result is None
    assert cache.get_stats().total_queries == 0


async def _empty_search(*_args, **_kwargs):
    return [], []
