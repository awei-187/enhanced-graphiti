import argparse
import asyncio
import contextvars
import functools
import inspect
import json
import logging
import os
import sys
from collections import defaultdict
from logging import INFO
from pathlib import Path
from time import perf_counter

# Allow running this file directly (without installing the package)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_STRUCTURED_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

# Configure Ollama LLM client
llm_config = LLMConfig(
    api_key=os.environ.get('OPENAI_API_KEY'),
    base_url=os.environ.get('OPENAI_BASE_URL'),
)

llm_client = OpenAIGenericClient(config=llm_config)


_CURRENT_QUERY_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'current_query_key', default=None
)


def _wrap_timed_async_method(
    obj: object,
    method_name: str,
    interval_sink: dict[str, list[tuple[float, float]]],
) -> None:
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return

    if not inspect.iscoroutinefunction(method):
        return

    @functools.wraps(method)
    async def _wrapped(*args, **kwargs):
        key = _CURRENT_QUERY_KEY.get()
        start = perf_counter()
        try:
            return await method(*args, **kwargs)
        finally:
            end = perf_counter()
            if key is not None:
                interval_sink.setdefault(key, []).append((start, end))

    setattr(obj, method_name, _wrapped)


def _union_duration_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = intervals_sorted[0]
    for start, end in intervals_sorted[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return sum(end - start for start, end in merged)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Graphiti search on (my)LoCoMo dataset with selectable cache strategy.'
    )
    default_dataset = (
        Path(__file__).resolve().parent / 'data' / 'locomo10_expanded.json'
    )
    default_output = (
        Path(__file__).resolve().parent / 'data' / 'graphiti_locomo_search_results.json'
    )
    default_stats_output = (
        Path(__file__).resolve().parent / 'data' / 'graphiti_locomo_cache_stats.json'
    )
    default_latency_output = (
        Path(__file__).resolve().parent / 'data' / 'graphiti_locomo_latency_stats.json'
    )

    parser.add_argument(
        '--dataset',
        type=Path,
        default=default_dataset,
        help=f'Dataset JSON path (default: {default_dataset})',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=default_output,
        help=f'Where to write search contexts (default: {default_output})',
    )
    parser.add_argument(
        '--cache-stats-output',
        type=Path,
        default=default_stats_output,
        help=f'Where to write cache stats JSON (default: {default_stats_output})',
    )
    parser.add_argument(
        '--latency-stats-output',
        type=Path,
        default=default_latency_output,
        help=f'Where to write latency stats JSON (default: {default_latency_output})',
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        '--resume',
        dest='resume',
        action='store_true',
        default=True,
        help='Resume from existing output files when possible (default).',
    )
    resume_group.add_argument(
        '--restart',
        dest='resume',
        action='store_false',
        help='Ignore existing output files and start a fresh run.',
    )

    parser.add_argument(
        '--cache-strategy',
        choices=['direct', 'semantic', 'subgraph'],
        default='direct',
        help=(
            'Search strategy: direct (no cache), semantic (semantic cache only), '
            'subgraph (subgraph cache only). Default: semantic.'
        ),
    )
    # Backward-compatible flags for existing scripts.
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        '--enable-cache',
        dest='legacy_enable_cache',
        action='store_true',
        default=None,
        help='[Deprecated] Alias of --cache-strategy semantic.',
    )
    cache_group.add_argument(
        '--disable-cache',
        dest='legacy_enable_cache',
        action='store_false',
        help='[Deprecated] Alias of --cache-strategy direct.',
    )
    parser.add_argument(
        '--cache-threshold',
        type=float,
        default=0.9,
        help='Cosine similarity threshold for cache hit (default: 0.9).',
    )
    parser.add_argument(
        '--cache-ttl-seconds',
        type=float,
        default=3600.0,
        help='Cache entry TTL in seconds (default: 3600).',
    )
    parser.add_argument(
        '--cache-max-entries',
        type=int,
        default=1000,
        help='Max cache entries (default: 1000).',
    )
    parser.add_argument(
        '--cache-index-type',
        choices=['auto', 'faiss_exact', 'faiss_hnsw', 'numpy'],
        default='auto',
        help='Cache index type (default: auto).',
    )
    parser.add_argument(
        '--cache-hnsw-m',
        type=int,
        default=32,
        help='HNSW graph degree when using faiss_hnsw (default: 32).',
    )
    parser.add_argument(
        '--cache-hnsw-ef-search',
        type=int,
        default=64,
        help='HNSW ef_search when using faiss_hnsw (default: 64).',
    )
    parser.add_argument(
        '--cache-hnsw-ef-construction',
        type=int,
        default=200,
        help='HNSW ef_construction when using faiss_hnsw (default: 200).',
    )
    parser.add_argument(
        '--subgraph-cache-ttl-seconds',
        type=float,
        default=3600.0,
        help='Subgraph cache entry TTL in seconds (default: 3600).',
    )
    parser.add_argument(
        '--subgraph-cache-max-entries',
        type=int,
        default=100,
        help='Max subgraph cache entries (default: 100).',
    )
    parser.add_argument(
        '--subgraph-cache-min-top1-score',
        type=float,
        default=0.75,
        help='Min normalized top-1 score to accept subgraph cache hit (default: 0.75).',
    )
    parser.add_argument(
        '--subgraph-cache-max-candidates',
        type=int,
        default=50,
        help='Max local candidates reranked from cached subgraph (default: 50).',
    )
    parser.add_argument(
        '--enable-structured-search',
        action='store_true',
        help='Add query-aware structured graph search as an extra edge retrieval channel.',
    )
    parser.add_argument(
        '--enable-single-entity-structured-search',
        action='store_true',
        help='Allow structured search to expand adjacent edges when only one query entity links.',
    )
    parser.add_argument(
        '--single-entity-structured-limit',
        type=int,
        default=3,
        help='Max one-entity structured edge candidates to add before reranking (default: 3).',
    )
    args = parser.parse_args()
    if args.legacy_enable_cache is not None:
        args.cache_strategy = 'semantic' if args.legacy_enable_cache else 'direct'
    return args


def _init_graphiti(args: argparse.Namespace) -> Graphiti:
    interval_sink: dict[str, list[tuple[float, float]]] = {}

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_BASE_URL'),
        )
    )
    # Expose sink so main() can compute per-query model-wait union time.
    setattr(embedder, '_timing_sink', interval_sink)
    cross_encoder = OpenAIRerankerClient(config=llm_config)
    # In practice, search latency mostly waits on embedder + reranker.
    _wrap_timed_async_method(embedder, 'create', interval_sink)
    _wrap_timed_async_method(embedder, 'create_batch', interval_sink)
    _wrap_timed_async_method(cross_encoder, 'rank', interval_sink)
    _wrap_timed_async_method(llm_client, 'generate_response', interval_sink)

    enable_semantic_cache = args.cache_strategy == 'semantic'
    enable_subgraph_cache = args.cache_strategy == 'subgraph'

    return Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        enable_cache=enable_semantic_cache,
        cache_similarity_threshold=args.cache_threshold,
        cache_ttl_seconds=args.cache_ttl_seconds,
        cache_max_entries=args.cache_max_entries,
        cache_index_type=args.cache_index_type,
        cache_hnsw_m=args.cache_hnsw_m,
        cache_hnsw_ef_search=args.cache_hnsw_ef_search,
        cache_hnsw_ef_construction=args.cache_hnsw_ef_construction,
        enable_subgraph_cache=enable_subgraph_cache,
        subgraph_cache_ttl_seconds=args.subgraph_cache_ttl_seconds,
        subgraph_cache_max_entries=args.subgraph_cache_max_entries,
        subgraph_cache_min_top1_score=args.subgraph_cache_min_top1_score,
        subgraph_cache_max_candidates=args.subgraph_cache_max_candidates,
    )


def _validate_output_paths(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    cache_out = args.cache_stats_output.resolve()
    latency_out = args.latency_stats_output.resolve()

    if out == cache_out:
        raise ValueError(
            '--output and --cache-stats-output point to the same file; cache stats would overwrite search results.'
        )
    if out == latency_out:
        raise ValueError(
            '--output and --latency-stats-output point to the same file; latency stats would overwrite search results.'
        )
    if cache_out == latency_out:
        raise ValueError(
            '--cache-stats-output and --latency-stats-output point to the same file; pick two different paths.'
        )

#################################################
# HELPER FUNCTIONS (与Zep脚本完全相同)
#################################################
TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""

def compose_search_context(edges: list[EntityEdge], nodes: list[EntityNode]) -> str:
    facts = [f'  - {edge.fact} (event_time: {edge.valid_at})' for edge in edges]
    entities = [f'  - {node.name}: {node.summary}' for node in nodes]
    return TEMPLATE.format(facts='\n'.join(facts), entities='\n'.join(entities))


def _load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Cannot resume because {path.resolve()} is not valid JSON') from exc


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'{path.name}.tmp')
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    temp_path.replace(path)


def _latency_payload(latency_rows: list[dict[str, object]]) -> dict[str, object]:
    total_latency_ms = [
        row['total_latency_ms']
        for row in latency_rows
        if isinstance(row.get('total_latency_ms'), int | float)
    ]
    absolute_latency_ms = [
        row['absolute_latency_ms']
        for row in latency_rows
        if isinstance(row.get('absolute_latency_ms'), int | float)
    ]
    avg_total_ms = (
        sum(total_latency_ms) / len(total_latency_ms) if total_latency_ms else 0.0
    )
    avg_absolute_ms = (
        sum(absolute_latency_ms) / len(absolute_latency_ms) if absolute_latency_ms else 0.0
    )
    return {
        'queries_evaluated': len(latency_rows),
        'avg_total_latency_ms': avg_total_ms,
        'avg_absolute_latency_ms': avg_absolute_ms,
        'rows': latency_rows,
    }


def _load_checkpoint(
    args: argparse.Namespace,
) -> tuple[defaultdict[str, list[dict[str, object]]], list[dict[str, object]]]:
    graphiti_search_results: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    latency_rows: list[dict[str, object]] = []

    if not args.resume:
        return graphiti_search_results, latency_rows

    saved_results = _load_json_file(args.output, {})
    if not isinstance(saved_results, dict):
        raise ValueError(f'Cannot resume because {args.output.resolve()} is not a JSON object')
    for owner_id, rows in saved_results.items():
        if not isinstance(rows, list):
            raise ValueError(
                f'Cannot resume because results for owner_id={owner_id!r} are not a list'
            )
        graphiti_search_results[str(owner_id)].extend(rows)

    saved_latency = _load_json_file(args.latency_stats_output, {})
    if isinstance(saved_latency, dict):
        rows = saved_latency.get('rows', [])
        if isinstance(rows, list):
            latency_rows = rows
    elif saved_latency:
        raise ValueError(
            f'Cannot resume because {args.latency_stats_output.resolve()} is not a JSON object'
        )

    completed = sum(len(rows) for rows in graphiti_search_results.values())
    if completed or latency_rows:
        logger.info(
            'Resuming from checkpoint: completed_results=%d latency_rows=%d',
            completed,
            len(latency_rows),
        )

    return graphiti_search_results, latency_rows


def _save_checkpoint(
    args: argparse.Namespace,
    graphiti_search_results: defaultdict[str, list[dict[str, object]]],
    latency_rows: list[dict[str, object]],
) -> None:
    _atomic_write_json(args.output, dict(graphiti_search_results))
    _atomic_write_json(args.latency_stats_output, _latency_payload(latency_rows))


async def main():
    args = _parse_args()
    _validate_output_paths(args)
    graphiti = _init_graphiti(args)
    try:
        # Load LOCOMO JSON data
        dataset_path = args.dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f'Dataset not found: {dataset_path.resolve()}')
        locomo_data = json.loads(dataset_path.read_text(encoding='utf-8'))

        graphiti_search_results, latency_rows = _load_checkpoint(args)

        # Use a single combined search config (edges + nodes) so cache semantics are consistent.
        edge_recipe = (
            EDGE_HYBRID_SEARCH_STRUCTURED_CROSS_ENCODER
            if args.enable_structured_search
            else EDGE_HYBRID_SEARCH_CROSS_ENCODER
        )
        combined_search_config = SearchConfig(
            edge_config=edge_recipe.edge_config.model_copy(deep=True)
            if edge_recipe.edge_config
            else None,
            node_config=NODE_HYBRID_SEARCH_RRF.node_config.model_copy(deep=True)
            if NODE_HYBRID_SEARCH_RRF.node_config
            else None,
            episode_config=None,
            community_config=None,
            limit=20,
        )
        if combined_search_config.edge_config is not None:
            combined_search_config.edge_config.structured_allow_single_entity = (
                args.enable_single_entity_structured_search
            )
            combined_search_config.edge_config.structured_single_entity_limit = (
                args.single_entity_structured_limit
            )
        
        # 遍历所有用户/对话
        for conversation in locomo_data:
            # 在 Graphiti 中，我们使用 owner_id 来实现用户隔离
            owner_id = str(conversation.get('sample_id'))
            qa_set = conversation.get('qa', [])
            completed_for_owner = len(graphiti_search_results[owner_id])
            seen_valid_for_owner = 0
            
            logger.info(f"--- Processing searches for Owner ID: {owner_id} ---")

            for qa in qa_set:
                query = qa.get('question')
                if qa.get('category') == 5 or not query:
                    continue
                if seen_valid_for_owner < completed_for_owner:
                    seen_valid_for_owner += 1
                    logger.info(
                        'Skipping completed query owner_id=%s completed_index=%d',
                        owner_id,
                        seen_valid_for_owner,
                    )
                    continue

                query_key = f'{owner_id}::{len(latency_rows)}'
                token = _CURRENT_QUERY_KEY.set(query_key)
                start_total = perf_counter()

                # --- 核心修改：只调用一次 Graphiti.search_() (edges + nodes) ---
                try:
                    search_results = await graphiti.search_(
                        query=query,
                        config=combined_search_config,
                        group_ids=[owner_id],
                    )

                    edges = search_results.edges
                    nodes = search_results.nodes

                    context = compose_search_context(edges, nodes)

                    end_total = perf_counter()
                finally:
                    _CURRENT_QUERY_KEY.reset(token)

                duration_total_s = end_total - start_total
                interval_sink = getattr(graphiti.embedder, '_timing_sink', {})
                intervals = interval_sink.get(query_key, []) if isinstance(interval_sink, dict) else []
                model_wait_s = _union_duration_seconds(intervals)
                duration_absolute_s = max(0.0, duration_total_s - model_wait_s)

                duration_ms = duration_total_s * 1000
                graphiti_search_results[owner_id].append(
                    {
                        'context': context,
                        'duration_ms': duration_ms,
                        'question': query,
                    }
                )

                latency_rows.append(
                    {
                        'owner_id': owner_id,
                        'question': query,
                        'total_latency_ms': duration_total_s * 1000,
                        'absolute_latency_ms': duration_absolute_s * 1000,
                        'model_wait_ms': model_wait_s * 1000,
                    }
                )
                seen_valid_for_owner += 1
                _save_checkpoint(args, graphiti_search_results, latency_rows)
                logger.info(
                    'Checkpoint saved owner_id=%s completed_for_owner=%d total_completed=%d',
                    owner_id,
                    seen_valid_for_owner,
                    len(latency_rows),
                )

            logger.info(f"Finished all searches for Owner ID: {owner_id}")

        # 保存结果到文件
        _atomic_write_json(args.output, dict(graphiti_search_results))
        logger.info(f'Successfully saved Graphiti search results to {args.output.resolve()}')

        # 保存延迟统计到文件
        _atomic_write_json(args.latency_stats_output, _latency_payload(latency_rows))
        logger.info(f'Saved latency stats to {args.latency_stats_output.resolve()}')

        semantic_cache_stats = graphiti.get_cache_stats()
        subgraph_cache_stats = graphiti.get_subgraph_cache_stats()
        payload = {
            'cache_strategy': args.cache_strategy,
            'semantic_cache_enabled': args.cache_strategy == 'semantic',
            'subgraph_cache_enabled': args.cache_strategy == 'subgraph',
            'config': {
                'cache_similarity_threshold': args.cache_threshold,
                'cache_ttl_seconds': args.cache_ttl_seconds,
                'cache_max_entries': args.cache_max_entries,
                'cache_index_type': args.cache_index_type,
                'cache_hnsw_m': args.cache_hnsw_m,
                'cache_hnsw_ef_search': args.cache_hnsw_ef_search,
                'cache_hnsw_ef_construction': args.cache_hnsw_ef_construction,
                'subgraph_cache_ttl_seconds': args.subgraph_cache_ttl_seconds,
                'subgraph_cache_max_entries': args.subgraph_cache_max_entries,
                'subgraph_cache_min_top1_score': args.subgraph_cache_min_top1_score,
                'subgraph_cache_max_candidates': args.subgraph_cache_max_candidates,
                'structured_search_enabled': args.enable_structured_search,
                'single_entity_structured_search_enabled': (
                    args.enable_single_entity_structured_search
                ),
                'single_entity_structured_limit': args.single_entity_structured_limit,
            },
            'semantic_cache_stats': semantic_cache_stats,
            'subgraph_cache_stats': subgraph_cache_stats,
        }

        if semantic_cache_stats is None:
            logger.info('Semantic cache disabled; no cache stats available.')
        else:
            logger.info(
                'Semantic cache stats: %s',
                json.dumps(semantic_cache_stats, ensure_ascii=False),
            )
        if subgraph_cache_stats is None:
            logger.info('Subgraph cache disabled; no cache stats available.')
        else:
            logger.info(
                'Subgraph cache stats: %s',
                json.dumps(subgraph_cache_stats, ensure_ascii=False),
            )

        args.cache_stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.cache_stats_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        logger.info(f'Saved cache stats to {args.cache_stats_output.resolve()}')

    finally:
        await graphiti.close()
        logger.info('\nNeo4j connection closed.')


if __name__ == "__main__":
    asyncio.run(main())
