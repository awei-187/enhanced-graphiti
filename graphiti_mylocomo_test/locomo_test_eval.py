import os
import json
from collections import defaultdict
from time import time
import logging

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import asyncio

CATEGORY_MAP = {
    4: "Single-Hop QA",
    1: "Multi-Hop QA",
    3: "Open-Domain QA",
    2: "Temporal QA",
}

class Grade(BaseModel):
  is_correct: str = Field(description='CORRECT or WRONG')
  reasoning: str = Field(description='Explain why the answer is correct or incorrect.')

async def locomo_grader(llm_client, question: str, gold_answer: str, response: str) -> bool:
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    ACCURACY_PROMPT = f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You williolw23 be given the following data:
        (1) a question (posed by one user to another user), 
        (2) a ’gold’ (ground truth) answer, 
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    try:
        response = await llm_client.beta.chat.completions.parse(
            model='gpt-4o-mini',
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": ACCURACY_PROMPT}],
            response_format=Grade,
            temperature=0,
        )
        result = response.choices[0].message.parsed

        return result.is_correct.strip().lower() == 'correct'
    
    except Exception as e:
        logging.error(f"Error while grading question '{question}': {e}. Defaulting to WRONG.")
        return False


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    oai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # 1. 建立问题 -> 类别的映射，并计算每个类别的应有总数
    question_to_category_map = {}
    total_questions_per_category = defaultdict(int)
    with open('data/locomo10_expanded_shuffled1.json', 'r', encoding='utf-8') as f:
        locomo_data = json.load(f)
        for user_data in locomo_data:
            for qa in user_data.get('qa', []):
                category = qa.get('category')
                if category in CATEGORY_MAP:
                    question_to_category_map[qa['question']] = category
                    total_questions_per_category[category] += 1
    
    expected_total_questions = sum(total_questions_per_category.values())
    logging.info(f"Found {expected_total_questions} total questions to evaluate across categories 1-4.")

    # --- 核心修改 1: 加载 Graphiti 的 response 文件 ---
    with open('data/graphiti_locomo_responses_no_cache1_enhanced3.json', 'r', encoding='utf-8') as file:
        graphiti_locomo_responses = json.load(file)

    # --- 核心修改 2: 定义 Graphiti 的 grades 文件路径 ---
    output_filepath = "data/graphiti_locomo_grades_no_cache1_enhanced3.json"
    graphiti_grades = defaultdict(list)

    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                loaded_grades = json.load(f)
                for key, value in loaded_grades.items():
                    graphiti_grades[key] = value
            logging.info(f"Successfully loaded existing grades from {output_filepath}. Resuming job.")
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON from {output_filepath}. Starting with a fresh run.")
            graphiti_grades = defaultdict(list)

    # --- 核心修改 3: 使用更健壮的循环方式 ---
    for owner_id, responses_to_grade in graphiti_locomo_responses.items():
        graded_items_for_owner = graphiti_grades.get(owner_id, [])
        graded_questions = {item['question'] for item in graded_items_for_owner}

        tasks_to_run = []
        for response in responses_to_grade:
            question = response.get('question')
            if response.get('golden_answer') is None or question in graded_questions:
                continue
            if question in question_to_category_map:
                tasks_to_run.append(response)

        if not tasks_to_run:
            logging.info(f"Owner {owner_id}: All relevant items already graded. Skipping.")
            continue
        
        logging.info(f"Owner {owner_id}: Found {len(tasks_to_run)} new items to grade.")

        tasks = [
            locomo_grader(oai_client, resp['question'], resp['golden_answer'], resp['answer'])
            for resp in tasks_to_run
        ]
        
        results = await asyncio.gather(*tasks)

        for response, grade in zip(tasks_to_run, results):
            question = response['question']
            category = question_to_category_map.get(question)
            graphiti_grades[owner_id].append({
                'question': question,
                'answer': response['answer'],
                'golden_answer': response['golden_answer'],
                'grade': grade,
                'category': category
            })

        os.makedirs("data", exist_ok=True)
        with open(output_filepath, "w", encoding='utf-8') as f:
            json.dump(dict(graphiti_grades), f, indent=2)
            logging.info(f"Successfully graded and saved progress for {owner_id}.")

    # --- 使用与 Zep 脚本完全相同的、健壮的计分逻辑 ---
    category_scores = defaultdict(int)
    
    for owner_id in graphiti_grades:
        for graded_item in graphiti_grades[owner_id]:
            category = graded_item.get('category')
            if category in CATEGORY_MAP and graded_item.get('grade'):
                category_scores[category] += 1

    print("\n" + "="*50)
    print("      GRAPHITI - DETAILED EVALUATION RESULTS")
    print("="*50)
    total_correct_score = 0
    
    for category_id, category_name in CATEGORY_MAP.items():
        score = category_scores[category_id]
        count = total_questions_per_category[category_id]
        total_correct_score += score
        accuracy = (score / count * 100) if count > 0 else 0
        print(f"{category_name:<20}: {score:4d} / {count:4d} | Accuracy: {accuracy:.2f}%")
    
    print("-"*50)
    
    overall_accuracy = (total_correct_score / expected_total_questions * 100) if expected_total_questions > 0 else 0
    print(f"{'OVERALL ACCURACY':<20}: {total_correct_score:4d} / {expected_total_questions:4d} | Accuracy: {overall_accuracy:.2f}%")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())