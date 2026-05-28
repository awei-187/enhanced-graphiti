"""
Structured graph search for query-aware edge retrieval.

This module extracts a lightweight query frame, links mentioned entities to graph
nodes, and retrieves candidate relationship edges constrained by those entities.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.llm_client import LLMClient
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode, get_entity_node_from_record
from graphiti_core.prompts.models import Message
from graphiti_core.graph_queries import get_vector_cosine_func_query
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)


class QueryFrame(BaseModel):
    """Structured interpretation of a user query."""

    entities: list[str] = Field(
        default_factory=list,
        description=(
            'Explicit entity names mentioned in the query. Include people, places, '
            'objects, events, and organizations.'
        ),
    )
    relation: str | None = Field(
        default=None,
        description='Natural language relation or predicate being asked about, if present.',
    )
    answer_slot: str | None = Field(
        default=None,
        description=(
            'What the query is asking for, such as time, relation, subject, object, '
            'location, or attribute.'
        ),
    )


def query_frame_prompt(query: str) -> list[Message]:
    return [
        Message(
            role='system',
            content=(
                'You extract a compact graph query frame from a question. '
                'Return only entities explicitly mentioned or unambiguously named in the question. '
                'Use a short natural-language relation phrase. If the question asks for one '
                'missing part of a likely subject-relation-object fact, describe that missing '
                'part in answer_slot.'
            ),
        ),
        Message(
            role='user',
            content=f"""
Question:
{query}

Examples:
- "When did Caroline go to the LGBTQ support group?"
  entities=["Caroline", "LGBTQ support group"], relation="go to / attend", answer_slot="time"
- "Who did Caroline go camping with?"
  entities=["Caroline"], relation="go camping with", answer_slot="object entity"
- "What is Caroline's relationship with Melanie?"
  entities=["Caroline", "Melanie"], relation="relationship between", answer_slot="relation"
""",
        ),
    ]


async def extract_query_frame(llm_client: LLMClient, query: str) -> QueryFrame | None:
    """Use the configured LLM client to extract a query frame."""
    try:
        response = await llm_client.generate_response(
            query_frame_prompt(query),
            response_model=QueryFrame,
            prompt_name='structured_search.query_frame',
        )
        frame = QueryFrame.model_validate(response)
        frame.entities = [entity.strip() for entity in frame.entities if entity.strip()]
        if not frame.entities and not frame.relation:
            return None
        return frame
    except Exception as exc:  # pragma: no cover - defensive fallback for retrieval
        logger.warning('Structured query extraction failed: %s', exc)
        return None


async def link_query_entities(
    driver: GraphDriver,
    entity_names: list[str],
    group_ids: list[str] | None,
    per_entity_limit: int,
) -> dict[str, list[EntityNode]]:
    """Link query entity names to graph nodes with exact and fuzzy name matching."""
    if not entity_names:
        return {}

    filter_query = ''
    params: dict[str, object] = {'entity_names': entity_names, 'limit': per_entity_limit}
    if group_ids is not None:
        filter_query = 'AND n.group_id IN $group_ids'
        params['group_ids'] = group_ids

    query = (
        """
        UNWIND $entity_names AS entity_name
        MATCH (n:Entity)
        WHERE (
            toLower(n.name) = toLower(entity_name)
            OR toLower(n.name) CONTAINS toLower(entity_name)
            OR toLower(entity_name) CONTAINS toLower(n.name)
        )
        """
        + filter_query
        + """
        WITH entity_name, n,
            CASE
                WHEN toLower(n.name) = toLower(entity_name) THEN 0
                WHEN toLower(n.name) CONTAINS toLower(entity_name) THEN 1
                ELSE 2
            END AS rank
        ORDER BY entity_name, rank, size(n.name)
        WITH entity_name, collect(n)[..$limit] AS nodes
        UNWIND nodes AS n
        RETURN entity_name,
        """
        + get_entity_node_return_query(driver.provider)
    )

    records, _, _ = await driver.execute_query(query, routing_='r', **params)

    linked: dict[str, list[EntityNode]] = {entity_name: [] for entity_name in entity_names}
    for record in records:
        entity_name = record['entity_name']
        linked.setdefault(entity_name, []).append(
            get_entity_node_from_record(record, driver.provider)
        )

    return linked


async def structured_edge_search(
    driver: GraphDriver,
    frame: QueryFrame,
    query_vector: list[float],
    relation_vector: list[float] | None,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = 10,
    per_entity_limit: int = 3,
    allow_single_entity: bool = False,
    single_entity_limit: int = 5,
) -> list[EntityEdge]:
    """Retrieve edges constrained by entities extracted from the query."""
    linked_entities = await link_query_entities(
        driver, frame.entities, group_ids, per_entity_limit
    )
    linked_entity_count = sum(1 for nodes in linked_entities.values() if nodes)
    candidate_node_uuids = [
        node.uuid for nodes in linked_entities.values() for node in nodes
    ]
    if not candidate_node_uuids:
        return []
    if linked_entity_count < 2 and not allow_single_entity:
        return []
    if linked_entity_count < 2 and not frame.relation:
        return []

    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )
    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' AND ' + (' AND '.join(filter_queries))

    search_vector = relation_vector or query_vector
    params = {
        'node_uuids': candidate_node_uuids,
        'relation': (frame.relation or '').strip().lower(),
        'search_vector': search_vector,
        'limit': min(limit, single_entity_limit) if linked_entity_count < 2 else limit,
        **filter_params,
    }

    if driver.provider == GraphProvider.KUZU:
        edge_pattern = '(n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)'
        search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'
    else:
        edge_pattern = '(n:Entity)-[e:RELATES_TO]->(m:Entity)'
        search_vector_var = '$search_vector'

    if linked_entity_count >= 2:
        match_query = f"""
        MATCH {edge_pattern}
        WHERE n.uuid IN $node_uuids AND m.uuid IN $node_uuids AND n.uuid <> m.uuid
        """
    else:
        match_query = f"""
        MATCH {edge_pattern}
        WHERE (n.uuid IN $node_uuids OR m.uuid IN $node_uuids)
        """

    query = (
        match_query
        + filter_query
        + """
        WITH DISTINCT e, n, m,
        """
        + get_vector_cosine_func_query('e.fact_embedding', search_vector_var, driver.provider)
        + """ AS vector_score,
            CASE
                WHEN $relation <> ''
                    AND (
                        toLower(coalesce(e.fact, '')) CONTAINS $relation
                        OR toLower(coalesce(e.name, '')) CONTAINS $relation
                    )
                THEN 1
                ELSE 0
            END AS relation_match
        ORDER BY relation_match DESC, vector_score DESC
        LIMIT $limit
        RETURN
        """
        + get_entity_edge_return_query(driver.provider)
    )

    records, _, _ = await driver.execute_query(query, routing_='r', **params)
    edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

    logger.debug(
        'Structured edge search frame=%s linked_entities=%s returned=%d',
        frame.model_dump(),
        {name: [node.name for node in nodes] for name, nodes in linked_entities.items()},
        len(edges),
    )
    return edges
