import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging import INFO

from pathlib import Path

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
from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ingest (my)LoCoMo conversation turns into Neo4j via Graphiti.')
    default_dataset = Path(__file__).resolve().parent / 'data' / 'locomo10_expanded.json'
    default_progress_dir = Path(__file__).resolve().parent / 'progress_logs'
    parser.add_argument(
        '--dataset',
        type=Path,
        default=default_dataset,
        help=f'Dataset JSON path (default: {default_dataset})',
    )
    parser.add_argument(
        '--progress-dir',
        type=Path,
        default=default_progress_dir,
        help=f'Progress log directory (default: {default_progress_dir})',
    )
    return parser.parse_args()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

llm_config = LLMConfig(
    api_key=os.environ.get('OPENAI_API_KEY'),
    base_url=os.environ.get('OPENAI_BASE_URL'),
)

llm_client = OpenAIGenericClient(config=llm_config)

graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_BASE_URL'),
        )
    ),
    cross_encoder=OpenAIRerankerClient(config=llm_config),
)

# --- 新增常量，与 zep 脚本保持一致 ---
MAX_SESSION_COUNT = 35

# --- 辅助函数 ---
def load_progress(progress_file):
    if not os.path.exists(progress_file):
        return set()
    logger.info(f"Loading progress from '{progress_file}'...")
    with open(progress_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def save_progress_batch(progress_file, processed_ids):
    with open(progress_file, 'a', encoding='utf-8') as f:
        for pid in processed_ids:
            f.write(f"{pid}\n")

async def process_batch(batch, owner_id, progress_file):
    if not batch:
        return 0

    tasks = [
        graphiti.add_episode(
            name=f'conv_{owner_id}_turn_{item["turn"]["dia_id"]}',
            episode_body=item["episode_body"],
            source=EpisodeType.text,
            source_description='locomo10_dialogue_turn',
            reference_time=item["reference_time"],
            group_id=owner_id,
        ) for item in batch
    ]
    
    try:
        await asyncio.gather(*tasks)
        processed_ids = [item['turn']['dia_id'] for item in batch]
        save_progress_batch(progress_file, processed_ids)
        logger.info(f"Owner '{owner_id}': Successfully processed a batch of {len(batch)} turns.")
        return len(batch)
    except Exception as e:
        logger.error(f"Owner '{owner_id}': An error occurred while processing a batch: {e}. Aborting this owner.", exc_info=True)
        raise

async def main():
    args = _parse_args()
    try:
        await graphiti.build_indices_and_constraints()

        dataset_path = args.dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f'Dataset not found: {dataset_path.resolve()}')
        locomo_data = json.loads(dataset_path.read_text(encoding='utf-8'))
        
        for conversation_to_process in locomo_data:
            owner_id = conversation_to_process['sample_id']
            conversation_content = conversation_to_process['conversation']
            
            args.progress_dir.mkdir(parents=True, exist_ok=True)
            progress_file = str(args.progress_dir / f'progress_{owner_id}.log')
            processed_dia_ids = load_progress(progress_file)
            
            logger.info(f"--- Processing Owner ID: {owner_id} ---")
            logger.info(f"Found {len(processed_dia_ids)} turns already processed for this owner.")

            batch_size = 20
            batch = []
            newly_added_counter = 0

            try:
                # ###############################################################
                # ### MODIFICATION START: 仿照 zep-cloud 脚本的迭代逻辑 ###
                # ###############################################################
                for session_idx in range(MAX_SESSION_COUNT):
                    session_key = f'session_{session_idx}'
                    session_turns = conversation_content.get(session_key)

                    if session_turns is None:
                        continue

                    datetime_key = f"{session_key}_date_time"
                    datetime_str = conversation_content.get(datetime_key)
                    reference_time = datetime.now(timezone.utc)
                    if datetime_str:
                        try:
                            session_date_with_utc = datetime_str + ' UTC'
                            date_format = '%I:%M %p on %d %B, %Y UTC'
                            reference_time = datetime.strptime(session_date_with_utc, date_format).replace(tzinfo=timezone.utc)
                        except ValueError:
                            logger.warning(f"Owner '{owner_id}': Could not parse date '{datetime_str}'. Using current time.")
                    
                    for turn in session_turns:
                        if 'text' not in turn or not turn['text']:
                            continue
                        
                        dia_id = turn['dia_id']
                        if dia_id in processed_dia_ids:
                            continue
                        
                        blip_caption = turn.get('blip_caption')
                        img_description = f" (description of attached image: {blip_caption})" if blip_caption else ''
                        
                        episode_body = f"{turn.get('speaker')}: {turn.get('text')}{img_description}"

                        batch.append({
                            "turn": turn, 
                            "episode_body": episode_body,
                            "reference_time": reference_time
                        })
                        
                        if len(batch) >= batch_size:
                            count = await process_batch(batch, owner_id, progress_file)
                            newly_added_counter += count
                            batch = []
                # ###############################################################
                # ### MODIFICATION END ###
                # ###############################################################

                if batch:
                    count = await process_batch(batch, owner_id, progress_file)
                    newly_added_counter += count

                if newly_added_counter == 0 and len(processed_dia_ids) > 0:
                    logger.info(f"Owner '{owner_id}': No new episodes to add. Processing complete for this owner.")
                else:
                    logger.info(f"Owner '{owner_id}': Successfully added {newly_added_counter} new turns.")

            except Exception:
                logger.error(f"Stopping processing for owner '{owner_id}' due to a batch processing failure.")
                continue

    finally:
        await graphiti.close()
        logger.info('\nNeo4j connection closed.')


if __name__ == '__main__':
    asyncio.run(main())