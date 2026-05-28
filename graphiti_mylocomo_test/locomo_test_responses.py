import os
import json
from collections import defaultdict
from time import time
import logging

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio

# --- 和之前版本相同的核心函数 ---
async def locomo_response(llm_client, context: str, question: str) -> str:
    system_prompt = """
        You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.
        """

    prompt = f"""
    # CONTEXT:
    You have access to facts and entities from a conversation.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. Always convert relative time references to specific dates, months, or years.
    6. Be as specific as possible when talking about people, places, and events
    7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
    Clarification:
    When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.
    
    Example:
    
    Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
    Question: What day did I go to the vet?
    Correct Answer: March 15, 2023
    Explanation:
    Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.


    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Context:

    {context}

    Question: {question}
    Answer:
    """
    
    # 注意：我们将 API 调用包裹在 try-except 中，以便重试逻辑可以捕获错误
    try:
        response = await llm_client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": prompt}],
                    temperature=0,
                )
        result = response.choices[0].message.content or ''
        return result
    except Exception as e:
        # 抛出异常，让重试包装器捕获它
        raise e

# ###############################################################
# ### MODIFICATION START: 新增带指数退避的重试包装器 ###
# ###############################################################
async def locomo_response_with_retry(llm_client, context: str, question: str, max_retries=5) -> str:
    """
    调用 locomo_response，并在失败时自动重试。
    """
    last_exception = None
    # 初始延迟时间为 2 秒
    delay = 2.0
    for attempt in range(max_retries):
        try:
            # 尝试调用原始函数
            return await locomo_response(llm_client, context, question)
        except Exception as e:
            last_exception = e
            logging.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for question '{question}'. "
                f"Error: {e}. Retrying in {delay:.1f} seconds..."
            )
            # 等待一段时间
            await asyncio.sleep(delay)
            # 增加下一次的延迟时间 (指数退避)
            delay *= 2
    
    # 如果所有重试都失败了
    logging.error(
        f"All {max_retries} retries failed for question '{question}'. "
        f"Final error: {last_exception}"
    )
    return f"Error: Failed to generate response after {max_retries} retries."
# ###############################################################
# ### MODIFICATION END ###
# ###############################################################

async def process_qa(qa_item, search_context, oai_client):
    start = time()
    query = qa_item.get('question')
    gold_answer = qa_item.get('answer')

    # --- 核心修改：调用新的、带重试功能的函数 ---
    generated_answer = await locomo_response_with_retry(oai_client, search_context, query)

    duration_ms = (time() - start) * 1000

    return {'question': query, 'answer': generated_answer, 'golden_answer': gold_answer, 'duration_ms': duration_ms}


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    oai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    with open('data/graphiti_mylocomo_search_no_cache1_enhanced3.json', 'r', encoding='utf-8') as file:
        graphiti_search_results = json.load(file)
    
    with open('data/locomo10_expanded_shuffled1.json', 'r', encoding='utf-8') as f:
        locomo_data = json.load(f)

    output_filepath = "data/graphiti_locomo_responses_no_cache1_enhanced3.json"
    graphiti_responses = {}

    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                graphiti_responses = json.load(f)
            logging.info(f"Successfully loaded existing results from {output_filepath}. Resuming job.")
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON from {output_filepath}. Starting with a fresh run.")
            graphiti_responses = {}

    for conversation in locomo_data:
        owner_id = conversation.get('sample_id')
        if not owner_id:
            continue

        qa_set = [qa for qa in conversation.get('qa', []) if qa.get('category') != 5]
        search_results_for_owner = graphiti_search_results.get(owner_id, [])
        search_context_map = {res['question']: res['context'] for res in search_results_for_owner}
        existing_responses_for_owner = graphiti_responses.get(owner_id, [])
        answered_questions = {resp['question'] for resp in existing_responses_for_owner}

        tasks_to_run = []
        for qa_item in qa_set:
            question = qa_item.get('question')
            if question not in answered_questions and question in search_context_map:
                search_context = search_context_map[question]
                tasks_to_run.append(process_qa(qa_item, search_context, oai_client))

        if not tasks_to_run:
            logging.info(f"Owner {owner_id}: All questions already answered. Skipping.")
            continue
        
        logging.info(f"Owner {owner_id}: Found {len(tasks_to_run)} new questions to process.")

        new_responses = await asyncio.gather(*tasks_to_run)
        
        if owner_id not in graphiti_responses:
            graphiti_responses[owner_id] = []
        graphiti_responses[owner_id].extend(new_responses)

        os.makedirs("data", exist_ok=True)
        with open(output_filepath, "w", encoding='utf-8') as f:
            json.dump(graphiti_responses, f, indent=2)
            logging.info(f"Successfully processed and saved progress for {owner_id}.")


if __name__ == "__main__":
    asyncio.run(main())