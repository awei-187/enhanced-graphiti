"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from time import time
from typing import TYPE_CHECKING

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.errors import SearchRerankerError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

# Import cache module (avoid circular imports)
if TYPE_CHECKING:
    from graphiti_core.cache import SemanticCache
    from graphiti_core.cache import SubgraphCache
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    maximal_marginal_relevance,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
    rrf_weighted,
)
from graphiti_core.search.structured_search import (
    QueryFrame,
    extract_query_frame,
    structured_edge_search,
)

logger = logging.getLogger(__name__)


def _cross_encoder_candidate_uuids(
    search_result_uuids: list[list[str]],
    candidate_uuids: list[str],
    limit: int,
    min_score: float,
) -> list[str]:
    """Pre-rank retrieval candidates before sending a wider set to the cross encoder."""
    preliminary_uuids, _ = rrf(search_result_uuids, min_score=min_score)
    if not preliminary_uuids:
        preliminary_uuids = candidate_uuids
    return preliminary_uuids[: 3 * limit]


async def search(
    clients: GraphitiClients,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    query_vector: list[float] | None = None,
    driver: GraphDriver | None = None,
    # ========== Semantic Cache Parameters ==========
    cache: 'SemanticCache | None' = None,
    use_cache: bool = True,
    # ========== Subgraph Cache Parameters ==========
    subgraph_cache: 'SubgraphCache | None' = None,
    use_subgraph_cache: bool = False,
) -> SearchResults:
    """
    Perform a hybrid search on the knowledge graph with optional semantic caching.

    Parameters
    ----------
    clients : GraphitiClients
        The Graphiti clients containing driver, embedder, and cross_encoder.
    query : str
        The search query string.
    group_ids : list[str] | None
        The group IDs to filter search results.
    config : SearchConfig
        The search configuration.
    search_filter : SearchFilters
        Additional search filters.
    center_node_uuid : str | None
        Optional center node for distance-based reranking.
    bfs_origin_node_uuids : list[str] | None
        Optional origin nodes for BFS search.
    query_vector : list[float] | None
        Pre-computed query embedding vector.
    driver : GraphDriver | None
        Optional graph driver override.
    cache : SemanticCache | None
        Optional semantic cache instance. If None and use_cache is True,
        the global cache will be used.
    use_cache : bool
        Whether to use semantic caching. Defaults to True.

    Returns
    -------
    SearchResults
        The search results containing edges, nodes, episodes, and communities.
    """
    start = time()

    driver = driver or clients.driver
    embedder = clients.embedder
    cross_encoder = clients.cross_encoder

    if query.strip() == '':
        return SearchResults()

    # ========== Step 1: Determine if embedding is needed ==========
    needs_embedding = (
        config.edge_config
        and EdgeSearchMethod.cosine_similarity in config.edge_config.search_methods
        or config.edge_config
        and EdgeReranker.mmr == config.edge_config.reranker
        or config.node_config
        and NodeSearchMethod.cosine_similarity in config.node_config.search_methods
        or config.node_config
        and NodeReranker.mmr == config.node_config.reranker
        or (
            config.community_config
            and CommunitySearchMethod.cosine_similarity in config.community_config.search_methods
        )
        or (config.community_config and CommunityReranker.mmr == config.community_config.reranker)
        or use_cache  # Semantic cache needs embedding for similarity matching
        or (
            use_subgraph_cache and subgraph_cache is not None
        )  # Subgraph cache may need embedding for local similarity search
    )

    if needs_embedding:
        search_vector = (
            query_vector
            if query_vector is not None
            else await embedder.create(input_data=[query.replace('\n', ' ')])
        )
    else:
        search_vector = [0.0] * EMBEDDING_DIM

    # Normalize group_ids early for consistent cache key matching
    normalized_group_ids = group_ids if group_ids and group_ids != [''] else None

    # ========== Step 2: Check Subgraph Cache ==========
    # Subgraph cache is checked before semantic cache because it re-runs the configured
    # search against a cached local subgraph and can reject low-confidence node matches.
    if use_subgraph_cache and subgraph_cache is not None:
        subgraph_result = await subgraph_cache.get(
            query=query,
            query_vector=search_vector,
            group_ids=normalized_group_ids,
            config=config,
            search_filter=search_filter,
            cross_encoder=cross_encoder,
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
        )
        if subgraph_result is not None:
            return subgraph_result.result

    # ========== Step 3: Check Semantic Cache ==========
    # When subgraph cache is enabled, a subgraph miss or low-score rejection must fall
    # through to the database so the threshold fallback remains meaningful.
    if use_cache and not (use_subgraph_cache and subgraph_cache is not None):
        # Import here to avoid circular imports
        from graphiti_core.cache import get_semantic_cache

        if cache is None:
            cache = get_semantic_cache()

        # Try to get cached result
        cached_entry = cache.get(search_vector, normalized_group_ids)
        if cached_entry is not None:
            latency = (time() - start) * 1000
            logger.info(
                f'[CACHE HIT] query="{query[:50]}..." latency={latency:.2f}ms '
                f'hit_count={cached_entry.hit_count}'
            )
            return cached_entry.result

    # ========== Step 4: Execute Original Search Logic ==========
    group_ids = normalized_group_ids

    needs_query_frame = (
        config.edge_config
        and EdgeSearchMethod.structured in config.edge_config.search_methods
    )
    query_frame = (
        await extract_query_frame(clients.llm_client, query) if needs_query_frame else None
    )
    relation_vector = (
        await embedder.create(input_data=[query_frame.relation.replace('\n', ' ')])
        if query_frame is not None and query_frame.relation
        else None
    )

    (
        (edges, edge_reranker_scores),
        (nodes, node_reranker_scores),
        (episodes, episode_reranker_scores),
        (communities, community_reranker_scores),
    ) = await semaphore_gather(
        edge_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.edge_config,
            search_filter,
            query_frame,
            relation_vector,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        ),
        node_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.node_config,
            search_filter,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        ),
        episode_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.episode_config,
            search_filter,
            config.limit,
            config.reranker_min_score,
        ),
        community_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.community_config,
            config.limit,
            config.reranker_min_score,
        ),
    )

    results = SearchResults(
        edges=edges,
        edge_reranker_scores=edge_reranker_scores,
        nodes=nodes,
        node_reranker_scores=node_reranker_scores,
        episodes=episodes,
        episode_reranker_scores=episode_reranker_scores,
        communities=communities,
        community_reranker_scores=community_reranker_scores,
    )

    # ========== Step 5: Store Result in Cache ==========
    if use_cache and cache is not None:
        # Extract related entity names from results for invalidation tracking
        related_entities = {node.name for node in nodes}
        cache.put(query, search_vector, results, related_entities, group_ids)
        logger.debug(f'[CACHE PUT] query="{query[:50]}..." entities={len(related_entities)}')

    if use_subgraph_cache and subgraph_cache is not None:
        related_entities = {node.name for node in nodes}
        subgraph_cache.put(
            results,
            group_ids,
            config,
            search_filter,
            center_node_uuid,
            bfs_origin_node_uuids,
            related_entities=related_entities,
        )

    latency = (time() - start) * 1000

    # Update miss latency with the actual full search time
    if use_cache and cache is not None:
        cache.update_miss_latency(latency)

    logger.debug(f'search returned context for query {query} in {latency} ms')

    return results


async def edge_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: EdgeSearchConfig | None,
    search_filter: SearchFilters,
    query_frame: QueryFrame | None = None,
    relation_vector: list[float] | None = None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> tuple[list[EntityEdge], list[float]]:
    if config is None:
        return [], []

    structured_weight = 1.0
    if query_frame is not None:
        entity_count = len(query_frame.entities)
        if entity_count >= 2 and query_frame.relation:
            structured_weight = 2.0
        elif (entity_count == 1 and query_frame.relation) or (
            entity_count == 2 and not query_frame.relation
        ):
            structured_weight = 1.0
        else:
            structured_weight = 0.0

    # Build search tasks based on configured search methods
    search_tasks = []
    search_weights: list[float] = []
    if EdgeSearchMethod.bm25 in config.search_methods:
        search_tasks.append(
            edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
        )
        search_weights.append(1.0)
    if EdgeSearchMethod.cosine_similarity in config.search_methods:
        search_tasks.append(
            edge_similarity_search(
                driver,
                query_vector,
                None,
                None,
                search_filter,
                group_ids,
                2 * limit,
                config.sim_min_score,
            )
        )
        search_weights.append(1.0)
    if EdgeSearchMethod.bfs in config.search_methods:
        search_tasks.append(
            edge_bfs_search(
                driver,
                bfs_origin_node_uuids,
                config.bfs_max_depth,
                search_filter,
                group_ids,
                2 * limit,
            )
        )
        search_weights.append(1.0)
    if EdgeSearchMethod.structured in config.search_methods:
        if query_frame is not None:
            search_tasks.append(
                structured_edge_search(
                    driver,
                    query_frame,
                    query_vector,
                    relation_vector,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.structured_max_entity_candidates,
                    config.structured_allow_single_entity,
                    config.structured_single_entity_limit,
                )
            )
            search_weights.append(structured_weight)

    # Execute only the configured search methods
    search_results: list[list[EntityEdge]] = []
    if search_tasks:
        search_results = list(await semaphore_gather(*search_tasks))

    if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
        source_node_uuids = [edge.source_node_uuid for result in search_results for edge in result]
        search_results.append(
            await edge_bfs_search(
                driver,
                source_node_uuids,
                config.bfs_max_depth,
                search_filter,
                group_ids,
                2 * limit,
            )
        )

    edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}

    reranked_uuids: list[str] = []
    edge_scores: list[float] = []
    if config.reranker == EdgeReranker.rrf or config.reranker == EdgeReranker.episode_mentions:
        search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

        if search_weights and len(search_weights) == len(search_result_uuids):
            reranked_uuids, edge_scores = rrf_weighted(
                search_result_uuids, search_weights, min_score=reranker_min_score
            )
        else:
            reranked_uuids, edge_scores = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == EdgeReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_edges(
            driver, list(edge_uuid_map.values())
        )
        reranked_uuids, edge_scores = maximal_marginal_relevance(
            query_vector,
            search_result_uuids_and_vectors,
            config.mmr_lambda,
            reranker_min_score,
        )
    elif config.reranker == EdgeReranker.cross_encoder:
        cross_encoder_uuids = _cross_encoder_candidate_uuids(
            [[edge.uuid for edge in result] for result in search_results],
            list(edge_uuid_map.keys()),
            limit,
            min_score=reranker_min_score,
        )
        fact_to_uuid_map = {
            edge_uuid_map[uuid].fact: uuid for uuid in cross_encoder_uuids if uuid in edge_uuid_map
        }
        if not fact_to_uuid_map:
            return [], []
        reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
        reranked_uuids = [
            fact_to_uuid_map[fact] for fact, score in reranked_facts if score >= reranker_min_score
        ]
        edge_scores = [score for _, score in reranked_facts if score >= reranker_min_score]
    elif config.reranker == EdgeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')

        # use rrf as a preliminary sort
        sorted_result_uuids, node_scores = rrf(
            [[edge.uuid for edge in result] for result in search_results],
            min_score=reranker_min_score,
        )
        sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

        # node distance reranking
        source_to_edge_uuid_map = defaultdict(list)
        for edge in sorted_results:
            source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

        source_uuids = [source_node_uuid for source_node_uuid in source_to_edge_uuid_map]

        reranked_node_uuids, edge_scores = await node_distance_reranker(
            driver, source_uuids, center_node_uuid, min_score=reranker_min_score
        )

        for node_uuid in reranked_node_uuids:
            reranked_uuids.extend(source_to_edge_uuid_map[node_uuid])

    reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]

    if config.reranker == EdgeReranker.episode_mentions:
        reranked_edges.sort(reverse=True, key=lambda edge: len(edge.episodes))

    return reranked_edges[:limit], edge_scores[:limit]


async def node_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: NodeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> tuple[list[EntityNode], list[float]]:
    if config is None:
        return [], []

    # Build search tasks based on configured search methods
    search_tasks = []
    if NodeSearchMethod.bm25 in config.search_methods:
        search_tasks.append(
            node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
        )
    if NodeSearchMethod.cosine_similarity in config.search_methods:
        search_tasks.append(
            node_similarity_search(
                driver,
                query_vector,
                search_filter,
                group_ids,
                2 * limit,
                config.sim_min_score,
            )
        )
    if NodeSearchMethod.bfs in config.search_methods:
        search_tasks.append(
            node_bfs_search(
                driver,
                bfs_origin_node_uuids,
                search_filter,
                config.bfs_max_depth,
                group_ids,
                2 * limit,
            )
        )

    # Execute only the configured search methods
    search_results: list[list[EntityNode]] = []
    if search_tasks:
        search_results = list(await semaphore_gather(*search_tasks))

    if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
        origin_node_uuids = [node.uuid for result in search_results for node in result]
        search_results.append(
            await node_bfs_search(
                driver,
                origin_node_uuids,
                search_filter,
                config.bfs_max_depth,
                group_ids,
                2 * limit,
            )
        )

    search_result_uuids = [[node.uuid for node in result] for result in search_results]
    node_uuid_map = {node.uuid: node for result in search_results for node in result}

    reranked_uuids: list[str] = []
    node_scores: list[float] = []
    if config.reranker == NodeReranker.rrf:
        reranked_uuids, node_scores = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == NodeReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_nodes(
            driver, list(node_uuid_map.values())
        )

        reranked_uuids, node_scores = maximal_marginal_relevance(
            query_vector,
            search_result_uuids_and_vectors,
            config.mmr_lambda,
            reranker_min_score,
        )
    elif config.reranker == NodeReranker.cross_encoder:
        cross_encoder_uuids = _cross_encoder_candidate_uuids(
            search_result_uuids,
            list(node_uuid_map.keys()),
            limit,
            min_score=reranker_min_score,
        )
        name_to_uuid_map = {
            node_uuid_map[uuid].name: uuid for uuid in cross_encoder_uuids if uuid in node_uuid_map
        }
        if not name_to_uuid_map:
            return [], []

        reranked_node_names = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
        reranked_uuids = [
            name_to_uuid_map[name]
            for name, score in reranked_node_names
            if score >= reranker_min_score
        ]
        node_scores = [score for _, score in reranked_node_names if score >= reranker_min_score]
    elif config.reranker == NodeReranker.episode_mentions:
        reranked_uuids, node_scores = await episode_mentions_reranker(
            driver, search_result_uuids, min_score=reranker_min_score
        )
    elif config.reranker == NodeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')
        reranked_uuids, node_scores = await node_distance_reranker(
            driver,
            rrf(search_result_uuids, min_score=reranker_min_score)[0],
            center_node_uuid,
            min_score=reranker_min_score,
        )

    reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_nodes[:limit], node_scores[:limit]


async def episode_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    _query_vector: list[float],
    group_ids: list[str] | None,
    config: EpisodeSearchConfig | None,
    search_filter: SearchFilters,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> tuple[list[EpisodicNode], list[float]]:
    if config is None:
        return [], []
    search_results: list[list[EpisodicNode]] = list(
        await semaphore_gather(
            *[
                episode_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
            ]
        )
    )

    search_result_uuids = [[episode.uuid for episode in result] for result in search_results]
    episode_uuid_map = {episode.uuid: episode for result in search_results for episode in result}

    reranked_uuids: list[str] = []
    episode_scores: list[float] = []
    if config.reranker == EpisodeReranker.rrf:
        reranked_uuids, episode_scores = rrf(search_result_uuids, min_score=reranker_min_score)

    elif config.reranker == EpisodeReranker.cross_encoder:
        # use rrf as a preliminary reranker
        cross_encoder_uuids = _cross_encoder_candidate_uuids(
            search_result_uuids,
            list(episode_uuid_map.keys()),
            limit,
            min_score=reranker_min_score,
        )
        rrf_results = [
            episode_uuid_map[uuid] for uuid in cross_encoder_uuids if uuid in episode_uuid_map
        ]

        content_to_uuid_map = {episode.content: episode.uuid for episode in rrf_results}
        if not content_to_uuid_map:
            return [], []

        reranked_contents = await cross_encoder.rank(query, list(content_to_uuid_map.keys()))
        reranked_uuids = [
            content_to_uuid_map[content]
            for content, score in reranked_contents
            if score >= reranker_min_score
        ]
        episode_scores = [score for _, score in reranked_contents if score >= reranker_min_score]

    reranked_episodes = [episode_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_episodes[:limit], episode_scores[:limit]


async def community_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> tuple[list[CommunityNode], list[float]]:
    if config is None:
        return [], []

    search_tasks = []
    if CommunitySearchMethod.bm25 in config.search_methods:
        search_tasks.append(community_fulltext_search(driver, query, group_ids, 2 * limit))
    if CommunitySearchMethod.cosine_similarity in config.search_methods:
        search_tasks.append(
            community_similarity_search(
                driver, query_vector, group_ids, 2 * limit, config.sim_min_score
            )
        )

    search_results: list[list[CommunityNode]] = []
    if search_tasks:
        search_results = list(await semaphore_gather(*search_tasks))

    search_result_uuids = [[community.uuid for community in result] for result in search_results]
    community_uuid_map = {
        community.uuid: community for result in search_results for community in result
    }

    reranked_uuids: list[str] = []
    community_scores: list[float] = []
    if config.reranker == CommunityReranker.rrf:
        reranked_uuids, community_scores = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == CommunityReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_communities(
            driver, list(community_uuid_map.values())
        )

        reranked_uuids, community_scores = maximal_marginal_relevance(
            query_vector, search_result_uuids_and_vectors, config.mmr_lambda, reranker_min_score
        )
    elif config.reranker == CommunityReranker.cross_encoder:
        cross_encoder_uuids = _cross_encoder_candidate_uuids(
            search_result_uuids,
            list(community_uuid_map.keys()),
            limit,
            min_score=reranker_min_score,
        )
        name_to_uuid_map = {
            community_uuid_map[uuid].name: uuid
            for uuid in cross_encoder_uuids
            if uuid in community_uuid_map
        }
        if not name_to_uuid_map:
            return [], []
        reranked_nodes = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
        reranked_uuids = [
            name_to_uuid_map[name] for name, score in reranked_nodes if score >= reranker_min_score
        ]
        community_scores = [score for _, score in reranked_nodes if score >= reranker_min_score]

    reranked_communities = [community_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_communities[:limit], community_scores[:limit]
