"""
Copyright 2024, Zep Software, Inc.

Subgraph cache for Graphiti search results.

The cache stores complete Graphiti objects returned by a full graph search and
reuses them as a local subgraph for later related queries. A cached subgraph is
accepted only when local reranking produces a sufficiently strong top-1 score;
otherwise callers should fall back to the full graph search.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Any, Iterable

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_config import (
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.search.search_utils import (
    calculate_cosine_similarity,
    maximal_marginal_relevance,
    rrf,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass
class SubgraphCacheStats:
    """Statistics for subgraph cache behavior."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallbacks_by_low_score: int = 0
    puts: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    invalidations_by_ttl: int = 0
    invalidations_by_entity: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    def record_hit(self, latency_ms: float) -> None:
        self.total_queries += 1
        self.cache_hits += 1
        self.avg_hit_latency_ms = (
            (self.avg_hit_latency_ms * (self.cache_hits - 1) + latency_ms)
            / self.cache_hits
        )

    def record_miss(self, latency_ms: float, low_score: bool = False) -> None:
        self.total_queries += 1
        self.cache_misses += 1
        if low_score:
            self.fallbacks_by_low_score += 1
        self.avg_miss_latency_ms = (
            (self.avg_miss_latency_ms * (self.cache_misses - 1) + latency_ms)
            / self.cache_misses
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f'{self.hit_rate:.2%}',
            'fallbacks_by_low_score': self.fallbacks_by_low_score,
            'puts': self.puts,
            'avg_hit_latency_ms': f'{self.avg_hit_latency_ms:.2f}',
            'avg_miss_latency_ms': f'{self.avg_miss_latency_ms:.2f}',
            'invalidations_by_ttl': self.invalidations_by_ttl,
            'invalidations_by_entity': self.invalidations_by_entity,
        }


@dataclass
class SubgraphSearchResult:
    """A local subgraph search result and its confidence."""

    result: SearchResults
    top1_score: float
    top2_score: float
    margin_score: float
    accept_score: float
    threshold: float
    cache_key: str
    node_top1_score: float
    confidence_score: float


@dataclass
class SubgraphCacheEntry:
    """A cached local graph containing complete Graphiti objects."""

    cache_key: str
    result: SearchResults
    group_ids: list[str] | None
    context_hash: str
    related_entities: set[str]
    created_at: Any = field(default_factory=utc_now)
    last_accessed_at: Any = field(default_factory=utc_now)
    hit_count: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        return (utc_now() - self.created_at).total_seconds() > ttl_seconds

    def touch(self) -> None:
        self.hit_count += 1
        self.last_accessed_at = utc_now()

    def matches_group_ids(self, group_ids: list[str] | None) -> bool:
        if self.group_ids is None and group_ids is None:
            return True
        if self.group_ids is None or group_ids is None:
            return False
        return set(self.group_ids) == set(group_ids)


class SubgraphCache:
    """
    Cache complete local subgraphs and search them before querying the full graph.

    The local search mirrors the main Graphiti search at a lightweight level:
    embedding cosine candidates, keyword candidates, and the configured reranker
    are applied to cached objects. The result is accepted only when a confidence
    score composed from top-1 and top1-top2 margin meets the configured threshold.
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 100,
        min_top1_score: float = 0.75,
        max_candidates: int = 50,
        enabled: bool = True,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.min_top1_score = min_top1_score
        self.max_candidates = max_candidates
        self.enabled = enabled
        self.min_top1_score_rrf = (
            (min_top1_score / (min_top1_score + 1.0)) if min_top1_score > 0 else 0.0
        )
        self.min_top1_score_mmr = min_top1_score
        self.min_top1_score_cross_encoder = min_top1_score
        self.cache_min_top1_score_rrf = min(self.min_top1_score_rrf, 0.4)
        self._entries: dict[str, SubgraphCacheEntry] = {}
        self._stats = SubgraphCacheStats()
        self._lock = Lock()

        logger.info(
            'SubgraphCache initialized: ttl=%ss, max_entries=%s, min_top1_score=%.3f',
            ttl_seconds,
            max_entries,
            min_top1_score,
        )

    async def get(
        self,
        query: str,
        query_vector: list[float],
        group_ids: list[str] | None,
        config: SearchConfig,
        search_filter: SearchFilters,
        cross_encoder: CrossEncoderClient,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
    ) -> SubgraphSearchResult | None:
        """Search compatible cached subgraphs and return a confident local result."""
        if not self.enabled:
            return None
        if self._requires_database_search(config):
            return None

        start = time()
        context_hash = self._context_hash(
            config, search_filter, center_node_uuid, bfs_origin_node_uuids
        )

        with self._lock:
            self._cleanup_expired()
            candidate_entries = [
                entry
                for entry in self._entries.values()
                if entry.context_hash == context_hash and entry.matches_group_ids(group_ids)
            ]

        if not candidate_entries:
            self._stats.record_miss((time() - start) * 1000)
            return None

        config_for_cache, swapped_cross_encoder = self._without_cross_encoder_rerankers(config)

        best: SubgraphSearchResult | None = None
        for entry in candidate_entries:
            local_result = await self._search_entry(
                entry,
                query,
                query_vector,
                config_for_cache,
                search_filter,
                cross_encoder,
                center_node_uuid,
                bfs_origin_node_uuids,
            )
            if local_result is None:
                continue
            if best is None or self._is_better_local_result(local_result, best):
                best = local_result

        latency_ms = (time() - start) * 1000
        if best is None:
            self._stats.record_miss(latency_ms)
            return None

        threshold = best.threshold
        if swapped_cross_encoder:
            threshold = min(threshold, self.cache_min_top1_score_rrf)

        if best.accept_score < threshold:
            self._stats.record_miss(latency_ms, low_score=True)
            logger.debug(
                '[SUBGRAPH CACHE FALLBACK] confidence_score=%.4f accept_score=%.4f '
                'top1=%.4f top2=%.4f margin=%.4f threshold=%.4f query="%s"',
                best.confidence_score,
                best.accept_score,
                best.top1_score,
                best.top2_score,
                best.margin_score,
                threshold,
                query[:50],
            )
            return None

        with self._lock:
            entry = self._entries.get(best.cache_key)
            if entry is not None:
                entry.touch()

        self._stats.record_hit(latency_ms)
        logger.info(
            '[SUBGRAPH CACHE HIT] query="%s..." latency=%.2fms confidence_score=%.4f '
            'accept_score=%.4f top1=%.4f top2=%.4f margin=%.4f threshold=%.4f',
            query[:50],
            latency_ms,
            best.confidence_score,
            best.accept_score,
            best.top1_score,
            best.top2_score,
            best.margin_score,
            threshold,
        )
        return best

    def put(
        self,
        result: SearchResults,
        group_ids: list[str] | None,
        config: SearchConfig,
        search_filter: SearchFilters,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        related_entities: set[str] | None = None,
    ) -> None:
        """Store a full graph result as a complete local subgraph."""
        if not self.enabled or self._is_empty_result(result):
            return

        context_hash = self._context_hash(
            config, search_filter, center_node_uuid, bfs_origin_node_uuids
        )
        related_entities = related_entities or self._extract_entities_from_results(result)
        object_uuids = self._object_uuids(result)
        cache_key = self._cache_key(group_ids, context_hash, related_entities, object_uuids)

        stored_result = result.model_copy(deep=True)
        entry = SubgraphCacheEntry(
            cache_key=cache_key,
            result=stored_result,
            group_ids=list(group_ids) if group_ids is not None else None,
            context_hash=context_hash,
            related_entities=related_entities,
        )

        with self._lock:
            if cache_key not in self._entries and len(self._entries) >= self.max_entries:
                self._evict_lru()
            self._entries[cache_key] = entry
            self._stats.puts += 1

        logger.debug(
            '[SUBGRAPH CACHE PUT] key=%s entities=%d objects=%d',
            cache_key[:12],
            len(related_entities),
            len(object_uuids),
        )

    def invalidate_by_entities(self, entity_names: set[str]) -> int:
        """Invalidate cached subgraphs related to updated entities."""
        if not self.enabled or not entity_names:
            return 0

        with self._lock:
            invalidated_keys = [
                key
                for key, entry in self._entries.items()
                if entry.related_entities & entity_names
            ]
            for key in invalidated_keys:
                self._entries.pop(key, None)
                self._stats.invalidations_by_entity += 1

        if invalidated_keys:
            logger.info(
                'Subgraph cache invalidated %d entries for entities: %s',
                len(invalidated_keys),
                list(entity_names)[:5],
            )
        return len(invalidated_keys)

    def invalidate_by_group_ids(self, group_ids: list[str]) -> int:
        if not self.enabled or not group_ids:
            return 0

        group_id_set = set(group_ids)
        with self._lock:
            invalidated_keys = [
                key
                for key, entry in self._entries.items()
                if entry.group_ids and set(entry.group_ids) & group_id_set
            ]
            for key in invalidated_keys:
                self._entries.pop(key, None)
        return len(invalidated_keys)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get_stats(self) -> SubgraphCacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._entries)

    async def _search_entry(
        self,
        entry: SubgraphCacheEntry,
        query: str,
        query_vector: list[float],
        config: SearchConfig,
        search_filter: SearchFilters,
        cross_encoder: CrossEncoderClient,
        center_node_uuid: str | None,
        bfs_origin_node_uuids: list[str] | None,
    ) -> SubgraphSearchResult | None:
        result = entry.result
        edges, edge_scores = await self._edge_search(
            result.edges,
            query,
            query_vector,
            config.edge_config,
            search_filter,
            cross_encoder,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        )
        nodes, node_scores = await self._node_search(
            result.nodes,
            result.edges,
            query,
            query_vector,
            config.node_config,
            search_filter,
            cross_encoder,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        )
        episodes, episode_scores = await self._episode_search(
            result.episodes,
            query,
            config.episode_config,
            search_filter,
            cross_encoder,
            config.limit,
            config.reranker_min_score,
        )
        communities, community_scores = await self._community_search(
            result.communities,
            query,
            query_vector,
            config.community_config,
            cross_encoder,
            config.limit,
            config.reranker_min_score,
        )

        local_result = SearchResults(
            edges=edges,
            edge_reranker_scores=edge_scores,
            nodes=nodes,
            node_reranker_scores=node_scores,
            episodes=episodes,
            episode_reranker_scores=episode_scores,
            communities=communities,
            community_reranker_scores=community_scores,
        )
        score_groups = [
            (edge_scores, config.edge_config.reranker if config.edge_config else None),
            (node_scores, config.node_config.reranker if config.node_config else None),
            (
                episode_scores,
                config.episode_config.reranker if config.episode_config else None,
            ),
            (
                community_scores,
                config.community_config.reranker if config.community_config else None,
            ),
        ]
        top1_score, top2_score, top1_reranker = self._top2_confidence(score_groups)
        node_top1_score, _, node_top1_reranker = self._top2_confidence(
            [(node_scores, config.node_config.reranker if config.node_config else None)]
        )
        if self._is_empty_result(local_result):
            return None
        use_node_confidence = config.node_config is not None and bool(node_scores)
        raw_confidence = node_top1_score if use_node_confidence else top1_score
        confidence_score = self._calibrate_confidence(raw_confidence, top1_score, top2_score, score_groups)
        active_reranker = node_top1_reranker if use_node_confidence else top1_reranker
        threshold = self._threshold_for_reranker(active_reranker)
        margin_score = max(0.0, top1_score - top2_score)
        accept_score = confidence_score
        return SubgraphSearchResult(
            local_result,
            top1_score,
            top2_score,
            margin_score,
            accept_score,
            threshold,
            entry.cache_key,
            node_top1_score,
            confidence_score,
        )

    def _is_better_local_result(
        self, candidate: SubgraphSearchResult, current: SubgraphSearchResult
    ) -> bool:
        if candidate.accept_score > current.accept_score:
            return True
        if candidate.accept_score < current.accept_score:
            return False
        return candidate.confidence_score > current.confidence_score

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        top1_score: float,
        top2_score: float,
        score_groups: list[tuple[list[float], Any]],
    ) -> float:
        if raw_confidence <= 0:
            return 0.0
        margin_score = max(0.0, top1_score - top2_score)
        # Keep confidence from saturating at 1.0 when ranking evidence is weak.
        separation_factor = 0.75 + 0.25 * margin_score
        nonempty_groups = sum(1 for scores, _reranker in score_groups if scores)
        support_factor = 0.9 + 0.1 * min(1.0, nonempty_groups / 2.0)
        singleton_penalty = 0.9 if top2_score <= 0 else 1.0
        calibrated = raw_confidence * separation_factor * support_factor * singleton_penalty
        return max(0.0, min(1.0, calibrated))

    async def _edge_search(
        self,
        edges: list[EntityEdge],
        query: str,
        query_vector: list[float],
        config: EdgeSearchConfig | None,
        search_filter: SearchFilters,
        cross_encoder: CrossEncoderClient,
        center_node_uuid: str | None,
        bfs_origin_node_uuids: list[str] | None,
        limit: int,
        reranker_min_score: float,
    ) -> tuple[list[EntityEdge], list[float]]:
        if config is None:
            return [], []
        if EdgeSearchMethod.structured in config.search_methods:
            return [], []

        candidates = [edge for edge in edges if self._edge_matches_filter(edge, search_filter)]
        edge_map = {edge.uuid: edge for edge in candidates}
        rankings: list[list[str]] = []

        if EdgeSearchMethod.bm25 in config.search_methods:
            rankings.append(self._keyword_ranking(candidates, query, self._edge_text, 2 * limit))
        if EdgeSearchMethod.cosine_similarity in config.search_methods:
            rankings.append(
                self._embedding_ranking(
                    candidates,
                    query_vector,
                    lambda edge: edge.fact_embedding,
                    config.sim_min_score,
                    2 * limit,
                )
            )
        if EdgeSearchMethod.bfs in config.search_methods:
            rankings.append(self._edge_bfs_ranking(candidates, bfs_origin_node_uuids, 2 * limit))
        if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            source_node_uuids = [
                edge.source_node_uuid for ranking in rankings for edge in self._objects(edge_map, ranking)
            ]
            rankings.append(self._edge_bfs_ranking(candidates, source_node_uuids, 2 * limit))

        return await self._rerank_objects(
            edge_map,
            rankings,
            query,
            query_vector,
            config.reranker,
            config.mmr_lambda,
            reranker_min_score,
            limit,
            cross_encoder,
            lambda edge: edge.fact,
            lambda edge: edge.fact_embedding,
            center_node_uuid,
            cached_edges=candidates,
        )

    async def _node_search(
        self,
        nodes: list[EntityNode],
        edges: list[EntityEdge],
        query: str,
        query_vector: list[float],
        config: NodeSearchConfig | None,
        search_filter: SearchFilters,
        cross_encoder: CrossEncoderClient,
        center_node_uuid: str | None,
        bfs_origin_node_uuids: list[str] | None,
        limit: int,
        reranker_min_score: float,
    ) -> tuple[list[EntityNode], list[float]]:
        if config is None:
            return [], []

        candidates = [node for node in nodes if self._node_matches_filter(node, search_filter)]
        node_map = {node.uuid: node for node in candidates}
        rankings: list[list[str]] = []

        if NodeSearchMethod.bm25 in config.search_methods:
            rankings.append(self._keyword_ranking(candidates, query, self._node_text, 2 * limit))
        if NodeSearchMethod.cosine_similarity in config.search_methods:
            rankings.append(
                self._embedding_ranking(
                    candidates,
                    query_vector,
                    lambda node: node.name_embedding,
                    config.sim_min_score,
                    2 * limit,
                )
            )
        if NodeSearchMethod.bfs in config.search_methods:
            rankings.append(
                self._node_bfs_ranking(candidates, edges, bfs_origin_node_uuids, 2 * limit)
            )
        if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            origin_node_uuids = [
                node.uuid for ranking in rankings for node in self._objects(node_map, ranking)
            ]
            rankings.append(self._node_bfs_ranking(candidates, edges, origin_node_uuids, 2 * limit))

        return await self._rerank_objects(
            node_map,
            rankings,
            query,
            query_vector,
            config.reranker,
            config.mmr_lambda,
            reranker_min_score,
            limit,
            cross_encoder,
            lambda node: node.name,
            lambda node: node.name_embedding,
            center_node_uuid,
            cached_edges=edges,
        )

    async def _episode_search(
        self,
        episodes: list[EpisodicNode],
        query: str,
        config: EpisodeSearchConfig | None,
        _search_filter: SearchFilters,
        cross_encoder: CrossEncoderClient,
        limit: int,
        reranker_min_score: float,
    ) -> tuple[list[EpisodicNode], list[float]]:
        if config is None:
            return [], []

        episode_map = {episode.uuid: episode for episode in episodes}
        rankings: list[list[str]] = []
        if EpisodeSearchMethod.bm25 in config.search_methods:
            rankings.append(self._keyword_ranking(episodes, query, self._episode_text, 2 * limit))

        return await self._rerank_objects(
            episode_map,
            rankings,
            query,
            [],
            config.reranker,
            config.mmr_lambda,
            reranker_min_score,
            limit,
            cross_encoder,
            lambda episode: episode.content,
            lambda _episode: None,
            None,
        )

    async def _community_search(
        self,
        communities: list[CommunityNode],
        query: str,
        query_vector: list[float],
        config: CommunitySearchConfig | None,
        cross_encoder: CrossEncoderClient,
        limit: int,
        reranker_min_score: float,
    ) -> tuple[list[CommunityNode], list[float]]:
        if config is None:
            return [], []

        community_map = {community.uuid: community for community in communities}
        rankings: list[list[str]] = []

        if CommunitySearchMethod.bm25 in config.search_methods:
            rankings.append(
                self._keyword_ranking(communities, query, self._community_text, 2 * limit)
            )
        if CommunitySearchMethod.cosine_similarity in config.search_methods:
            rankings.append(
                self._embedding_ranking(
                    communities,
                    query_vector,
                    lambda community: community.name_embedding,
                    config.sim_min_score,
                    2 * limit,
                )
            )

        return await self._rerank_objects(
            community_map,
            rankings,
            query,
            query_vector,
            config.reranker,
            config.mmr_lambda,
            reranker_min_score,
            limit,
            cross_encoder,
            lambda community: community.name,
            lambda community: community.name_embedding,
            None,
        )

    async def _rerank_objects(
        self,
        object_map: dict[str, Any],
        rankings: list[list[str]],
        query: str,
        query_vector: list[float],
        reranker: Any,
        mmr_lambda: float,
        reranker_min_score: float,
        limit: int,
        cross_encoder: CrossEncoderClient,
        text_getter: Any,
        embedding_getter: Any,
        center_node_uuid: str | None,
        cached_edges: list[EntityEdge] | None = None,
    ) -> tuple[list[Any], list[float]]:
        rankings = [ranking for ranking in rankings if ranking]
        if not rankings:
            return [], []

        candidate_uuids = list({uuid for ranking in rankings for uuid in ranking})
        reranked_uuids: list[str] = []
        scores: list[float] = []

        if reranker in {
            EdgeReranker.rrf,
            NodeReranker.rrf,
            EpisodeReranker.rrf,
            CommunityReranker.rrf,
        }:
            reranked_uuids, scores = rrf(rankings, min_score=reranker_min_score)
        elif reranker in {EdgeReranker.mmr, NodeReranker.mmr, CommunityReranker.mmr}:
            embeddings = {
                uuid: embedding_getter(object_map[uuid])
                for uuid in candidate_uuids
                if uuid in object_map and embedding_getter(object_map[uuid]) is not None
            }
            reranked_uuids, scores = maximal_marginal_relevance(
                query_vector, embeddings, mmr_lambda, reranker_min_score
            )
        elif reranker in {
            EdgeReranker.cross_encoder,
            NodeReranker.cross_encoder,
            EpisodeReranker.cross_encoder,
            CommunityReranker.cross_encoder,
        }:
            preselected_uuids = rrf(rankings, min_score=reranker_min_score)[0] or candidate_uuids
            preselected_uuids = preselected_uuids[: 3 * limit]
            text_to_uuid: dict[str, str] = {}
            texts: list[str] = []
            for uuid in preselected_uuids:
                if uuid not in object_map:
                    continue
                text = text_getter(object_map[uuid])
                if not text or text in text_to_uuid:
                    continue
                text_to_uuid[text] = uuid
                texts.append(text)
            if not texts:
                return [], []
            ranked_texts = await cross_encoder.rank(query, texts)
            for text, score in ranked_texts:
                if score >= reranker_min_score and text in text_to_uuid:
                    reranked_uuids.append(text_to_uuid[text])
                    scores.append(score)
        elif reranker in {EdgeReranker.node_distance, NodeReranker.node_distance}:
            if center_node_uuid is None or cached_edges is None:
                return [], []
            reranked_uuids, scores = self._node_distance_ranking(
                candidate_uuids, center_node_uuid, cached_edges, reranker_min_score
            )
        elif reranker in {EdgeReranker.episode_mentions, NodeReranker.episode_mentions}:
            mention_scores = self._episode_mention_scores(candidate_uuids, cached_edges or [])
            reranked_uuids = sorted(
                candidate_uuids, reverse=True, key=lambda uuid: mention_scores[uuid]
            )
            scores = [mention_scores[uuid] for uuid in reranked_uuids]
            reranked_uuids = [
                uuid for uuid, score in zip(reranked_uuids, scores) if score >= reranker_min_score
            ]
            scores = [score for score in scores if score >= reranker_min_score]

        reranked_objects = [object_map[uuid] for uuid in reranked_uuids if uuid in object_map]
        return reranked_objects[:limit], scores[:limit]

    def _keyword_ranking(
        self,
        objects: Iterable[Any],
        query: str,
        text_getter: Any,
        limit: int,
    ) -> list[str]:
        scored: list[tuple[str, float]] = []
        for obj in objects:
            score = self._keyword_score(query, text_getter(obj))
            if score > 0:
                scored.append((obj.uuid, score))
        scored.sort(reverse=True, key=lambda item: item[1])
        return [uuid for uuid, _score in scored[: min(limit, self.max_candidates)]]

    def _objects(self, object_map: dict[str, Any], uuids: list[str]) -> list[Any]:
        return [object_map[uuid] for uuid in uuids if uuid in object_map]

    def _embedding_ranking(
        self,
        objects: Iterable[Any],
        query_vector: list[float],
        embedding_getter: Any,
        min_score: float,
        limit: int,
    ) -> list[str]:
        scored: list[tuple[str, float]] = []
        for obj in objects:
            embedding = embedding_getter(obj)
            if embedding is None:
                continue
            score = calculate_cosine_similarity(query_vector, embedding)
            if score >= min_score:
                scored.append((obj.uuid, score))
        scored.sort(reverse=True, key=lambda item: item[1])
        return [uuid for uuid, _score in scored[: min(limit, self.max_candidates)]]

    def _edge_bfs_ranking(
        self,
        edges: list[EntityEdge],
        origins: list[str] | None,
        limit: int,
    ) -> list[str]:
        if not origins:
            return []
        origin_set = set(origins)
        matched = [
            edge.uuid
            for edge in edges
            if edge.source_node_uuid in origin_set or edge.target_node_uuid in origin_set
        ]
        return matched[:limit]

    def _node_bfs_ranking(
        self,
        nodes: list[EntityNode],
        edges: list[EntityEdge],
        origins: list[str] | None,
        limit: int,
    ) -> list[str]:
        if not origins:
            return []
        node_uuids = {node.uuid for node in nodes}
        adjacent = self._adjacency(edges)
        discovered: list[str] = []
        for origin in origins:
            for neighbor in adjacent.get(origin, set()):
                if neighbor in node_uuids and neighbor not in discovered:
                    discovered.append(neighbor)
        return discovered[:limit]

    def _node_distance_ranking(
        self,
        candidate_uuids: list[str],
        center_node_uuid: str,
        edges: list[EntityEdge],
        min_score: float,
    ) -> tuple[list[str], list[float]]:
        adjacent = self._adjacency(edges)
        distances = self._shortest_distances(center_node_uuid, adjacent)
        scored: list[tuple[str, float]] = []
        for uuid in candidate_uuids:
            if uuid == center_node_uuid:
                score = 10.0
            elif uuid in distances and distances[uuid] > 0:
                score = 1 / distances[uuid]
            else:
                continue
            if score >= min_score:
                scored.append((uuid, score))
        scored.sort(reverse=True, key=lambda item: item[1])
        return [uuid for uuid, _score in scored], [score for _uuid, score in scored]

    def _episode_mention_scores(
        self, candidate_uuids: list[str], edges: list[EntityEdge]
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for uuid in candidate_uuids:
            scores[uuid] = 0.0
        for edge in edges:
            episode_count = float(len(edge.episodes))
            if edge.source_node_uuid in scores:
                scores[edge.source_node_uuid] += episode_count
            if edge.target_node_uuid in scores:
                scores[edge.target_node_uuid] += episode_count
            if edge.uuid in scores:
                scores[edge.uuid] = max(scores[edge.uuid], episode_count)
        return scores

    def _top2_confidence(
        self, score_groups: list[tuple[list[float], Any]]
    ) -> tuple[float, float, Any]:
        pairs: list[tuple[float, Any]] = []
        for scores, reranker in score_groups:
            if not scores:
                continue
            top1 = self._normalize_score(scores[0], reranker)
            pairs.append((top1, reranker))
            if len(scores) > 1:
                top2 = self._normalize_score(scores[1], reranker)
                pairs.append((top2, reranker))
        if not pairs:
            return 0.0, 0.0, None
        pairs.sort(reverse=True, key=lambda item: item[0])
        top1_score, top1_reranker = pairs[0]
        top2_score = pairs[1][0] if len(pairs) > 1 else 0.0
        return top1_score, top2_score, top1_reranker

    def _threshold_for_reranker(self, reranker: Any) -> float:
        if reranker in {EdgeReranker.rrf, NodeReranker.rrf, EpisodeReranker.rrf, CommunityReranker.rrf}:
            return self.min_top1_score_rrf
        if reranker in {EdgeReranker.mmr, NodeReranker.mmr, CommunityReranker.mmr}:
            return self.min_top1_score_mmr
        if reranker in {
            EdgeReranker.cross_encoder,
            NodeReranker.cross_encoder,
            EpisodeReranker.cross_encoder,
            CommunityReranker.cross_encoder,
        }:
            return self.min_top1_score_cross_encoder
        return self.min_top1_score

    def _normalize_score(self, score: float, reranker: Any) -> float:
        if math.isnan(score):
            return 0.0
        if reranker in {
            EdgeReranker.cross_encoder,
            NodeReranker.cross_encoder,
            EpisodeReranker.cross_encoder,
            CommunityReranker.cross_encoder,
        }:
            if 0.0 <= score <= 1.0:
                return score
            return 1 / (1 + math.exp(-score))
        if reranker in {EdgeReranker.mmr, NodeReranker.mmr, CommunityReranker.mmr}:
            return max(0.0, min(1.0, (score + 1) / 2))
        if reranker in {EdgeReranker.node_distance, NodeReranker.node_distance}:
            return max(0.0, min(1.0, score / 10))
        if reranker in {EdgeReranker.episode_mentions, NodeReranker.episode_mentions}:
            return score / (score + 1) if score > 0 else 0.0
        # RRF and other unbounded positive scores: keep ordering but avoid saturating to 1.0.
        if score > 0:
            return score / (score + 1.0)
        return 0.0

    def _edge_matches_filter(self, edge: EntityEdge, search_filter: SearchFilters) -> bool:
        if search_filter.edge_uuids is not None and edge.uuid not in search_filter.edge_uuids:
            return False
        if search_filter.edge_types is not None and edge.name not in search_filter.edge_types:
            return False
        return (
            self._date_filters_pass(edge.valid_at, search_filter.valid_at)
            and self._date_filters_pass(edge.invalid_at, search_filter.invalid_at)
            and self._date_filters_pass(edge.created_at, search_filter.created_at)
            and self._date_filters_pass(edge.expired_at, search_filter.expired_at)
        )

    def _node_matches_filter(self, node: EntityNode, search_filter: SearchFilters) -> bool:
        if search_filter.node_labels is None:
            return True
        return set(search_filter.node_labels).issubset(set(node.labels))

    def _date_filters_pass(self, value: Any, filters: list[list[DateFilter]] | None) -> bool:
        if filters is None:
            return True
        for and_group in filters:
            if all(self._date_filter_pass(value, date_filter) for date_filter in and_group):
                return True
        return False

    def _date_filter_pass(self, value: Any, date_filter: DateFilter) -> bool:
        op = date_filter.comparison_operator
        if op == ComparisonOperator.is_null:
            return value is None
        if op == ComparisonOperator.is_not_null:
            return value is not None
        if value is None or date_filter.date is None:
            return False
        if op == ComparisonOperator.equals:
            return value == date_filter.date
        if op == ComparisonOperator.not_equals:
            return value != date_filter.date
        if op == ComparisonOperator.greater_than:
            return value > date_filter.date
        if op == ComparisonOperator.less_than:
            return value < date_filter.date
        if op == ComparisonOperator.greater_than_equal:
            return value >= date_filter.date
        if op == ComparisonOperator.less_than_equal:
            return value <= date_filter.date
        return False

    def _keyword_score(self, query: str, text: str) -> float:
        query_terms = self._tokens(query)
        if not query_terms:
            return 0.0
        text_terms = self._tokens(text)
        if not text_terms:
            return 0.0
        overlap = query_terms & text_terms
        score = len(overlap) / len(query_terms)
        if query.lower() in text.lower():
            score += 0.5
        return score

    def _tokens(self, text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text)}

    def _edge_text(self, edge: EntityEdge) -> str:
        return f'{edge.name} {edge.fact} {json.dumps(edge.attributes, default=str)}'

    def _node_text(self, node: EntityNode) -> str:
        return (
            f'{node.name} {node.summary} {" ".join(node.labels)} '
            f'{json.dumps(node.attributes, default=str)}'
        )

    def _episode_text(self, episode: EpisodicNode) -> str:
        return f'{episode.name} {episode.source_description} {episode.content}'

    def _community_text(self, community: CommunityNode) -> str:
        return f'{community.name} {community.summary}'

    def _adjacency(self, edges: list[EntityEdge]) -> dict[str, set[str]]:
        adjacent: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            adjacent[edge.source_node_uuid].add(edge.target_node_uuid)
            adjacent[edge.target_node_uuid].add(edge.source_node_uuid)
        return adjacent

    def _shortest_distances(
        self, origin: str, adjacent: dict[str, set[str]]
    ) -> dict[str, int]:
        distances = {origin: 0}
        queue: deque[str] = deque([origin])
        while queue:
            current = queue.popleft()
            for neighbor in adjacent.get(current, set()):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
        return distances

    def _extract_entities_from_results(self, result: SearchResults) -> set[str]:
        entities = {node.name for node in result.nodes if node.name}
        for community in result.communities:
            if community.name:
                entities.add(community.name)
        return entities

    def _requires_database_search(self, config: SearchConfig) -> bool:
        return (
            config.edge_config is not None
            and EdgeSearchMethod.structured in config.edge_config.search_methods
        )

    def _object_uuids(self, result: SearchResults) -> list[str]:
        return sorted(
            [
                *[edge.uuid for edge in result.edges],
                *[node.uuid for node in result.nodes],
                *[episode.uuid for episode in result.episodes],
                *[community.uuid for community in result.communities],
            ]
        )

    def _is_empty_result(self, result: SearchResults) -> bool:
        return not (result.edges or result.nodes or result.episodes or result.communities)

    def _without_cross_encoder_rerankers(self, config: SearchConfig) -> tuple[SearchConfig, bool]:
        updated = config.model_copy(deep=True)
        swapped = False
        if updated.edge_config and updated.edge_config.reranker == EdgeReranker.cross_encoder:
            updated.edge_config.reranker = EdgeReranker.rrf
            swapped = True
        if updated.node_config and updated.node_config.reranker == NodeReranker.cross_encoder:
            updated.node_config.reranker = NodeReranker.rrf
            swapped = True
        if updated.episode_config and updated.episode_config.reranker == EpisodeReranker.cross_encoder:
            updated.episode_config.reranker = EpisodeReranker.rrf
            swapped = True
        if updated.community_config and updated.community_config.reranker == CommunityReranker.cross_encoder:
            updated.community_config.reranker = CommunityReranker.rrf
            swapped = True
        return updated, swapped

    def _context_hash(
        self,
        config: SearchConfig,
        search_filter: SearchFilters,
        center_node_uuid: str | None,
        bfs_origin_node_uuids: list[str] | None,
    ) -> str:
        data = {
            'config': self._jsonable(config),
            'search_filter': self._jsonable(search_filter),
            'center_node_uuid': center_node_uuid,
            'bfs_origin_node_uuids': sorted(bfs_origin_node_uuids or []),
        }
        return self._hash(data)

    def _cache_key(
        self,
        group_ids: list[str] | None,
        context_hash: str,
        related_entities: set[str],
        object_uuids: list[str],
    ) -> str:
        return self._hash(
            {
                'group_ids': sorted(group_ids or []),
                'context_hash': context_hash,
                'related_entities': sorted(related_entities),
                'object_uuids': object_uuids[:100],
            }
        )

    def _jsonable(self, value: Any) -> Any:
        if hasattr(value, 'model_dump'):
            return value.model_dump(mode='json')
        if hasattr(value, 'dict'):
            return value.dict()
        return value

    def _hash(self, data: dict[str, Any]) -> str:
        payload = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _cleanup_expired(self) -> None:
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.is_expired(self.ttl_seconds)
        ]
        for key in expired_keys:
            self._entries.pop(key, None)
            self._stats.invalidations_by_ttl += 1

    def _evict_lru(self) -> None:
        if not self._entries:
            return
        lru_key = min(
            self._entries.keys(),
            key=lambda key: self._entries[key].last_accessed_at,
        )
        self._entries.pop(lru_key, None)
