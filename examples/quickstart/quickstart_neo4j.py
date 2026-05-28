"""
Copyright 2025, Zep Software, Inc.

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

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging - use DEBUG to see cache messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = '030412345678'

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neo4j connection
    # graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    try:
        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        #################################################
        # CACHE VALIDATION TEST
        #################################################
        # Test semantic cache by running same/similar
        # queries and measuring response times
        #################################################

        print("\n" + "=" * 50)
        print("CACHE VALIDATION TEST")
        print("=" * 50)

        # Debug: Check cache status
        print(f"\nCache enabled: {graphiti.enable_cache}")
        print(f"Cache object: {graphiti.cache}")
        if graphiti.cache:
            print(f"Cache size: {graphiti.cache.size}")
            print(f"Cache threshold: {graphiti.cache.similarity_threshold}")

        # Test 1: First query (Cache Miss expected)
        query1 = 'Who was the California Attorney General?'
        print(f"\n[Test 1] First query: '{query1}'")
        print("Expected: Cache MISS (first time seeing this query)")

        start_time = time.time()
        results1 = await graphiti.search(query1)
        elapsed1 = (time.time() - start_time) * 1000
        print(f"Results: {len(results1)} edges found")
        print(f"Time: {elapsed1:.2f} ms")

        # Test 2: Exact same query (Cache Hit expected)
        print(f"\n[Test 2] Same query again: '{query1}'")
        print("Expected: Cache HIT (exact same query)")

        start_time = time.time()
        results2 = await graphiti.search(query1)
        elapsed2 = (time.time() - start_time) * 1000
        print(f"Results: {len(results2)} edges found")
        print(f"Time: {elapsed2:.2f} ms")
        print(f"Speedup: {elapsed1 / elapsed2:.1f}x faster" if elapsed2 > 0 else "")

        # Test 3: Similar query (Cache Hit expected if similarity > threshold)
        query3 = 'Who is the Attorney General of California?'
        print(f"\n[Test 3] Similar query: '{query3}'")
        print("Expected: Cache HIT (semantically similar)")

        start_time = time.time()
        results3 = await graphiti.search(query3)
        elapsed3 = (time.time() - start_time) * 1000
        print(f"Results: {len(results3)} edges found")
        print(f"Time: {elapsed3:.2f} ms")

        # Test 4: Different query (Cache Miss expected)
        query4 = 'What is the capital of France?'
        print(f"\n[Test 4] Different query: '{query4}'")
        print("Expected: Cache MISS (different topic)")

        start_time = time.time()
        results4 = await graphiti.search(query4)
        elapsed4 = (time.time() - start_time) * 1000
        print(f"Results: {len(results4)} edges found")
        print(f"Time: {elapsed4:.2f} ms")

        # Print cache statistics
        print("\n" + "=" * 50)
        print("CACHE STATISTICS")
        print("=" * 50)
        cache_stats = graphiti.get_cache_stats()
        if cache_stats:
            print(f"Total Queries: {cache_stats['total_queries']}")
            print(f"Cache Hits:    {cache_stats['cache_hits']}")
            print(f"Cache Misses:  {cache_stats['cache_misses']}")
            print(f"Hit Rate:      {cache_stats['hit_rate']}")
            print(f"Avg Hit Latency:  {cache_stats['avg_hit_latency_ms']} ms")
            print(f"Avg Miss Latency: {cache_stats['avg_miss_latency_ms']} ms")
        else:
            print("Cache is disabled")
        print("=" * 50)

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
