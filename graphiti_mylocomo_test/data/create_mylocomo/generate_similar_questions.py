import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


INPUT_FILE = Path(__file__).resolve().parent / 'locomo10.json'
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / 'first10_similar_questions_test.json'
DEFAULT_EXPANDED_OUTPUT_FILE = Path(__file__).resolve().parent / 'locomo10_expanded.json'
DEFAULT_EXPAND_REPORT_FILE = Path(__file__).resolve().parent / 'locomo10_expand_report.json'
DEFAULT_CHECKPOINT_FILE = Path(__file__).resolve().parent / 'locomo10_expand_checkpoint.json'
PROGRESS_EVERY = 10


def log_progress(message: str) -> None:
    print(message, flush=True)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def ensure_question_mark(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text.endswith('?') else f'{text}?'


def rule_based_rewrite(question: str) -> str:
    q = question.strip()
    q_no_qmark = q[:-1] if q.endswith('?') else q
    lowered = q_no_qmark.lower()

    if lowered.startswith('when did '):
        return ensure_question_mark('At what time ' + q_no_qmark[5:].strip())
    if lowered.startswith('when is '):
        return ensure_question_mark('At what time ' + q_no_qmark[5:].strip())
    if lowered.startswith('what is '):
        return ensure_question_mark('Could you tell me what ' + q_no_qmark[8:].strip())
    if lowered.startswith('what are '):
        return ensure_question_mark('Could you tell me what ' + q_no_qmark[9:].strip())
    if lowered.startswith('what did '):
        return ensure_question_mark('Can you tell me what ' + q_no_qmark[9:].strip())
    if lowered.startswith('what does '):
        return ensure_question_mark('Can you tell me what ' + q_no_qmark[10:].strip())
    if lowered.startswith('why did '):
        return ensure_question_mark('For what reason ' + q_no_qmark[4:].strip())
    if lowered.startswith('how did '):
        return ensure_question_mark('In what way ' + q_no_qmark[4:].strip())
    if lowered.startswith('who '):
        return ensure_question_mark('Could you tell me who ' + q_no_qmark[4:].strip())

    return ensure_question_mark(f'Could you rephrase this: {q_no_qmark}')


def try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def call_chat_model_json(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None

    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    url = f'{base_url}/chat/completions'

    payload = {
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode('utf-8')
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    try:
        result = json.loads(body)
        content = result['choices'][0]['message']['content']
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        return None

    return try_parse_json(content)


def llm_rewrite(question: str, answer: str) -> str | None:
    system_prompt = (
        'You rewrite user questions. Keep meaning equivalent and keep the answer unchanged. '
        'Do not add new constraints, entities, times, negations, or assumptions. '
        'Return JSON only.'
    )
    user_prompt = (
        'Rewrite the question into one similar question with the same answer.\n'
        f'Original question: {question}\n'
        f'Expected answer: {answer}\n'
        'Output JSON format: {"similar_question":"..."}'
    )
    parsed = call_chat_model_json(system_prompt, user_prompt)
    if not parsed:
        return None
    candidate = parsed.get('similar_question')
    if not isinstance(candidate, str):
        return None
    return ensure_question_mark(candidate.strip())


def llm_validate_same_answer(original_question: str, answer: str, candidate_question: str) -> tuple[bool, str]:
    system_prompt = (
        'You validate whether two questions should have exactly the same answer. '
        'Be strict and conservative. Return JSON only.'
    )
    user_prompt = (
        f'Original question: {original_question}\n'
        f'Known correct answer: {answer}\n'
        f'Candidate similar question: {candidate_question}\n'
        'Return JSON: {"is_valid": true/false, "reason": "...", "similarity": 0.0}'
    )
    parsed = call_chat_model_json(system_prompt, user_prompt)
    if not parsed:
        return heuristic_validate_same_answer(original_question, answer, candidate_question)

    is_valid = bool(parsed.get('is_valid', False))
    reason = str(parsed.get('reason', 'no_reason'))
    return is_valid, reason


def tokenize(text: str) -> set[str]:
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))


def heuristic_validate_same_answer(
    original_question: str,
    _answer: str,
    candidate_question: str,
) -> tuple[bool, str]:
    o = normalize_text(original_question)
    c = normalize_text(candidate_question)

    if not c:
        return False, 'candidate_empty'
    if o == c:
        return False, 'candidate_same_as_original'

    o_tokens = tokenize(o)
    c_tokens = tokenize(c)
    if not o_tokens or not c_tokens:
        return False, 'tokenization_failed'

    overlap = len(o_tokens & c_tokens) / max(1, len(o_tokens | c_tokens))
    if overlap < 0.35:
        return False, f'low_overlap_{overlap:.2f}'

    return True, f'heuristic_pass_overlap_{overlap:.2f}'


def pick_answer(qa_item: dict[str, Any]) -> str:
    answer = qa_item.get('answer')
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    if answer is not None:
        return str(answer)

    adversarial_answer = qa_item.get('adversarial_answer')
    if isinstance(adversarial_answer, str) and adversarial_answer.strip():
        return adversarial_answer.strip()
    if adversarial_answer is not None:
        return str(adversarial_answer)

    return ''


def has_non_empty_question(qa_item: dict[str, Any]) -> bool:
    question = qa_item.get('question')
    return isinstance(question, str) and bool(question.strip())


def generate_one(question: str, answer: str) -> dict[str, Any]:
    candidate = rule_based_rewrite(question)
    is_valid, reason = llm_validate_same_answer(question, answer, candidate)
    method = 'rule'

    if not is_valid:
        llm_candidate = llm_rewrite(question, answer)
        if llm_candidate:
            second_valid, second_reason = llm_validate_same_answer(question, answer, llm_candidate)
            if second_valid:
                candidate = llm_candidate
                is_valid = True
                reason = second_reason
                method = 'llm_fallback'

    return {
        'original_question': question,
        'answer': answer,
        'generated_question': candidate,
        'method': method,
        'passed': is_valid,
        'validation_reason': reason,
    }


def collect_first_n_queries(data: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for conversation in data:
        sample_id = conversation.get('sample_id')
        qa_items = conversation.get('qa', [])
        if not isinstance(qa_items, list):
            continue

        for qa in qa_items:
            question = qa.get('question')
            if not isinstance(question, str) or not question.strip():
                continue

            collected.append(
                {
                    'sample_id': sample_id,
                    'category': qa.get('category'),
                    'question': question.strip(),
                    'answer': pick_answer(qa),
                }
            )

            if len(collected) >= limit:
                return collected

    return collected


def run_test(input_path: Path, output_path: Path, limit: int) -> dict[str, Any]:
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    queries = collect_first_n_queries(data, limit)
    total_queries = len(queries)
    log_progress(f'[test] start: total_queries={total_queries}, input={input_path}')

    results: list[dict[str, Any]] = []
    pass_count = 0
    for idx, item in enumerate(queries, 1):
        gen = generate_one(item['question'], item['answer'])
        row = {
            'sample_id': item['sample_id'],
            'category': item['category'],
            **gen,
        }
        results.append(row)
        if row['passed']:
            pass_count += 1

        if idx % PROGRESS_EVERY == 0 or idx == total_queries:
            progress = (idx / total_queries * 100) if total_queries else 100.0
            log_progress(
                f'[test] progress: {idx}/{total_queries} ({progress:.1f}%), '
                f'passed={pass_count}'
            )

    report = {
        'input_file': str(input_path),
        'tested_query_count': len(queries),
        'passed_count': pass_count,
        'pass_rate': (pass_count / len(queries)) if queries else 0.0,
        'uses_llm': bool(os.getenv('OPENAI_API_KEY')),
        'results': results,
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log_progress(
        f'[test] done: passed={pass_count}/{total_queries}, pass_rate={report["pass_rate"]:.4f}, '
        f'output={output_path}'
    )

    return report


def build_augmented_qa_item(qa_item: dict[str, Any], generated_question: str) -> dict[str, Any]:
    new_item = dict(qa_item)
    new_item['question'] = generated_question
    return new_item


def run_expand(
    input_path: Path,
    expanded_output_path: Path,
    report_output_path: Path,
    checkpoint_path: Path,
    resume: bool,
) -> dict[str, Any]:
    input_signature = compute_sha256(input_path)

    checkpoint_data: dict[str, Any] | None = None
    if resume and checkpoint_path.exists():
        try:
            with checkpoint_path.open('r', encoding='utf-8') as f:
                loaded = json.load(f)
            if (
                isinstance(loaded, dict)
                and loaded.get('input_file') == str(input_path)
                and loaded.get('input_signature') == input_signature
            ):
                checkpoint_data = loaded
        except json.JSONDecodeError:
            checkpoint_data = None

    start_sample_idx = 0
    total_original_questions = 0
    total_generated_questions = 0
    passed_count = 0
    rule_count = 0
    llm_fallback_count = 0
    failed_count = 0
    per_sample_stats: list[dict[str, Any]] = []

    if checkpoint_data and expanded_output_path.exists():
        with expanded_output_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        start_sample_idx = int(checkpoint_data.get('next_sample_idx', 0))
        total_original_questions = int(checkpoint_data.get('total_original_questions', 0))
        total_generated_questions = int(checkpoint_data.get('total_generated_questions', 0))
        passed_count = int(checkpoint_data.get('passed_count', 0))
        rule_count = int(checkpoint_data.get('rule_count', 0))
        llm_fallback_count = int(checkpoint_data.get('llm_fallback_count', 0))
        failed_count = int(checkpoint_data.get('failed_count', 0))
        loaded_stats = checkpoint_data.get('per_sample_stats', [])
        if isinstance(loaded_stats, list):
            per_sample_stats = loaded_stats
        log_progress(
            f'[expand] resume: next_sample_idx={start_sample_idx}, '
            f'processed_questions={total_original_questions}, checkpoint={checkpoint_path}'
        )
    else:
        with input_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if checkpoint_data and not expanded_output_path.exists():
            log_progress('[expand] checkpoint found but expanded output missing, starting from scratch.')

    with input_path.open('r', encoding='utf-8') as f:
        original_data = json.load(f)

    total_samples = len(original_data)
    total_questions_estimate = sum(
        1
        for conversation in original_data
        for qa_item in conversation.get('qa', [])
        if isinstance(qa_item, dict)
        and has_non_empty_question(qa_item)
    )
    log_progress(
        f'[expand] start: samples={total_samples}, estimated_questions={total_questions_estimate}, '
        f'input={input_path}, resume={resume}'
    )

    for sample_zero_idx in range(start_sample_idx, total_samples):
        conversation = data[sample_zero_idx]
        sample_idx = sample_zero_idx + 1
        sample_id = conversation.get('sample_id')
        qa_items = conversation.get('qa', [])
        if not isinstance(qa_items, list):
            log_progress(f'[expand] sample {sample_idx}/{total_samples} skipped: sample_id={sample_id}')
            per_sample_stats.append(
                {
                    'sample_id': sample_id,
                    'original_qa_count': 0,
                    'generated_qa_count': 0,
                    'pass_count': 0,
                    'fail_count': 0,
                }
            )

            checkpoint_payload = {
                'input_file': str(input_path),
                'input_signature': input_signature,
                'next_sample_idx': sample_zero_idx + 1,
                'total_original_questions': total_original_questions,
                'total_generated_questions': total_generated_questions,
                'passed_count': passed_count,
                'rule_count': rule_count,
                'llm_fallback_count': llm_fallback_count,
                'failed_count': failed_count,
                'per_sample_stats': per_sample_stats,
            }
            atomic_json_dump(data, expanded_output_path)
            atomic_json_dump(checkpoint_payload, checkpoint_path)
            continue

        log_progress(
            f'[expand] sample {sample_idx}/{total_samples} start: sample_id={sample_id}, '
            f'qa_count={len(qa_items)}'
        )

        new_qa_items: list[dict[str, Any]] = []
        sample_original_count = 0
        sample_generated_count = 0
        sample_pass_count = 0
        sample_fail_count = 0

        for qa_item in qa_items:
            new_qa_items.append(qa_item)

            question = qa_item.get('question')
            if not isinstance(question, str) or not question.strip():
                continue

            sample_original_count += 1
            total_original_questions += 1

            answer = pick_answer(qa_item)
            gen = generate_one(question.strip(), answer)

            if gen['passed']:
                new_qa_items.append(build_augmented_qa_item(qa_item, gen['generated_question']))
                sample_generated_count += 1
                total_generated_questions += 1
                sample_pass_count += 1
                passed_count += 1
                if gen['method'] == 'rule':
                    rule_count += 1
                elif gen['method'] == 'llm_fallback':
                    llm_fallback_count += 1
            else:
                sample_fail_count += 1
                failed_count += 1

            if total_original_questions % PROGRESS_EVERY == 0:
                progress = (
                    total_original_questions / total_questions_estimate * 100
                    if total_questions_estimate
                    else 100.0
                )
                log_progress(
                    f'[expand] progress: {total_original_questions}/{total_questions_estimate} '
                    f'({progress:.1f}%), generated={total_generated_questions}, '
                    f'failed={failed_count}, rule={rule_count}, llm_fallback={llm_fallback_count}'
                )

        conversation['qa'] = new_qa_items
        log_progress(
            f'[expand] sample {sample_idx}/{total_samples} done: sample_id={sample_id}, '
            f'generated={sample_generated_count}, failed={sample_fail_count}'
        )
        per_sample_stats.append(
            {
                'sample_id': sample_id,
                'original_qa_count': sample_original_count,
                'generated_qa_count': sample_generated_count,
                'pass_count': sample_pass_count,
                'fail_count': sample_fail_count,
            }
        )

        checkpoint_payload = {
            'input_file': str(input_path),
            'input_signature': input_signature,
            'next_sample_idx': sample_zero_idx + 1,
            'total_original_questions': total_original_questions,
            'total_generated_questions': total_generated_questions,
            'passed_count': passed_count,
            'rule_count': rule_count,
            'llm_fallback_count': llm_fallback_count,
            'failed_count': failed_count,
            'per_sample_stats': per_sample_stats,
        }
        atomic_json_dump(data, expanded_output_path)
        atomic_json_dump(checkpoint_payload, checkpoint_path)

    atomic_json_dump(data, expanded_output_path)

    report = {
        'input_file': str(input_path),
        'expanded_output_file': str(expanded_output_path),
        'total_original_questions': total_original_questions,
        'total_generated_questions': total_generated_questions,
        'pass_count': passed_count,
        'fail_count': failed_count,
        'pass_rate': (passed_count / total_original_questions) if total_original_questions else 0.0,
        'uses_llm': bool(os.getenv('OPENAI_API_KEY')),
        'rule_count': rule_count,
        'llm_fallback_count': llm_fallback_count,
        'per_sample_stats': per_sample_stats,
    }

    atomic_json_dump(report, report_output_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log_progress(f'[expand] checkpoint cleared: {checkpoint_path}')

    log_progress(
        f'[expand] done: generated={total_generated_questions}/{total_original_questions}, '
        f'pass_rate={report["pass_rate"]:.4f}, output={expanded_output_path}, report={report_output_path}'
    )

    return report


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    load_env_file(repo_root / '.env')
    load_env_file(script_dir / '.env')

    parser = argparse.ArgumentParser(
        description='Generate similar questions with same-answer validation for LOCOMO queries.'
    )
    parser.add_argument('--mode', choices=['test', 'expand'], default='test')
    parser.add_argument('--input', type=Path, default=INPUT_FILE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--expanded-output', type=Path, default=DEFAULT_EXPANDED_OUTPUT_FILE)
    parser.add_argument('--report-output', type=Path, default=DEFAULT_EXPAND_REPORT_FILE)
    parser.add_argument('--checkpoint-output', type=Path, default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    if args.mode == 'test':
        report = run_test(args.input, args.output, args.limit)
        print(json.dumps(
            {
                'mode': 'test',
                'tested_query_count': report['tested_query_count'],
                'passed_count': report['passed_count'],
                'pass_rate': report['pass_rate'],
                'uses_llm': report['uses_llm'],
                'output_file': str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    report = run_expand(
        args.input,
        args.expanded_output,
        args.report_output,
        args.checkpoint_output,
        resume=not args.no_resume,
    )
    print(json.dumps(
        {
            'mode': 'expand',
            'total_original_questions': report['total_original_questions'],
            'total_generated_questions': report['total_generated_questions'],
            'pass_rate': report['pass_rate'],
            'uses_llm': report['uses_llm'],
            'rule_count': report['rule_count'],
            'llm_fallback_count': report['llm_fallback_count'],
            'expanded_output_file': report['expanded_output_file'],
            'report_output_file': str(args.report_output),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == '__main__':
    main()
