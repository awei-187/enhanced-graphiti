"""
示例：如何将 SemanticCache 集成到 search 函数中

这个文件展示了修改后的 search 函数结构。
实际使用时，你需要将这些改动应用到 graphiti_core/search/search.py
"""

# ============================================================================
# 修改 1: 在 search/search.py 开头添加导入
# ============================================================================

"""
from graphiti_core.cache import SemanticCache, get_semantic_cache
"""


# ============================================================================
# 修改 2: 修改 search 函数，添加缓存逻辑
# ============================================================================

"""
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
    # ========== 新增参数 ==========
    cache: SemanticCache | None = None,
    use_cache: bool = True,
) -> SearchResults:
    start = time()

    driver = driver or clients.driver
    embedder = clients.embedder
    cross_encoder = clients.cross_encoder

    if query.strip() == '':
        return SearchResults()

    # ========== 步骤 1: 生成 Query Embedding ==========
    # 无论是否命中缓存，我们都需要 embedding
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
        or use_cache  # 缓存也需要 embedding
    )

    if needs_embedding:
        search_vector = (
            query_vector
            if query_vector is not None
            else await embedder.create(input_data=[query.replace('\\n', ' ')])
        )
    else:
        search_vector = [0.0] * EMBEDDING_DIM

    # ========== 步骤 2: 检查缓存 ==========
    if use_cache:
        if cache is None:
            cache = get_semantic_cache()

        cached_entry = cache.get(search_vector, group_ids)
        if cached_entry is not None:
            latency = (time() - start) * 1000
            logger.debug(f'Cache hit for query "{query}" in {latency} ms')
            return cached_entry.result

    # ========== 步骤 3: 执行原有搜索逻辑 ==========
    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids and group_ids != [''] else None

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

    # ========== 步骤 4: 存入缓存 ==========
    if use_cache and cache is not None:
        # 从结果中提取关联实体
        related_entities = {node.name for node in nodes}
        cache.put(query, search_vector, results, related_entities, group_ids)

    latency = (time() - start) * 1000
    logger.debug(f'search returned context for query {query} in {latency} ms')

    return results
"""


# ============================================================================
# 修改 3: 在 graphiti.py 的 add_episode 方法中添加缓存失效
# ============================================================================

"""
# 在 graphiti.py 的 add_episode 方法末尾（return 之前）添加：

# ========== 缓存失效：提取新增/更新的实体名称 ==========
updated_entity_names = {node.name for node in hydrated_nodes}
if updated_entity_names:
    from graphiti_core.cache import get_semantic_cache
    cache = get_semantic_cache()
    invalidated_count = cache.invalidate_by_entities(updated_entity_names)
    if invalidated_count > 0:
        logger.info(f'Invalidated {invalidated_count} cache entries for updated entities')
"""


# ============================================================================
# 修改 4: 在 Graphiti 类中添加缓存配置
# ============================================================================

"""
# 在 Graphiti.__init__ 中添加缓存配置参数：

def __init__(
    self,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    llm_client: LLMClient | None = None,
    embedder: EmbedderClient | None = None,
    cross_encoder: CrossEncoderClient | None = None,
    store_raw_episode_content: bool = True,
    graph_driver: GraphDriver | None = None,
    max_coroutines: int | None = None,
    tracer: Tracer | None = None,
    trace_span_prefix: str = 'graphiti',
    # ========== 新增缓存配置 ==========
    enable_cache: bool = True,
    cache_similarity_threshold: float = 0.9,
    cache_ttl_seconds: float = 300.0,
    cache_max_entries: int = 1000,
):
    # ... 原有初始化代码 ...

    # 初始化缓存
    if enable_cache:
        from graphiti_core.cache import SemanticCache
        self.cache = SemanticCache(
            similarity_threshold=cache_similarity_threshold,
            ttl_seconds=cache_ttl_seconds,
            max_entries=cache_max_entries,
            enabled=True,
        )
    else:
        self.cache = None
"""
