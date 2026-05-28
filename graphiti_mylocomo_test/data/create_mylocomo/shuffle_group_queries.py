"""Shuffle per-group query order for locomo10_expanded-style datasets.

This script shuffles the order of each group's `qa` list while leaving the
conversation and summaries untouched.

Default behavior writes a new file next to the input.

Examples
--------
# Deterministic shuffle (recommended for experiments)
python shuffle_group_queries.py --seed 42

# Specify input/output
python shuffle_group_queries.py --input locomo10_expanded.json --output locomo10_expanded_shuffled.json --seed 123

# Overwrite input file
python shuffle_group_queries.py --in-place --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Shuffle query (qa) order within each conversation group.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('locomo10_expanded.json'),
        help='Input JSON file (default: locomo10_expanded.json)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output JSON file (default: <input_stem>_shuffled.json)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible shuffling (default: None)',
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite input file (ignores --output if set).',
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _dump_json(path: Path, data: Any) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def shuffle_qa_in_each_group(dataset: list[dict[str, Any]], rng: random.Random) -> dict[str, int]:
    """Shuffle `qa` list within each group.

    Returns basic stats for logging.
    """
    group_count = 0
    total_qa = 0
    missing_qa = 0

    for group in dataset:
        group_count += 1
        qa = group.get('qa')
        if qa is None:
            missing_qa += 1
            continue
        if not isinstance(qa, list):
            raise TypeError(f'Expected group["qa"] to be a list, got {type(qa)}')
        total_qa += len(qa)
        rng.shuffle(qa)

    return {
        'groups': group_count,
        'total_qa': total_qa,
        'missing_qa_groups': missing_qa,
    }


def main() -> None:
    args = _parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f'Input not found: {input_path.resolve()}')

    output_path: Path
    if args.in_place:
        output_path = input_path
    else:
        output_path = (
            args.output
            if args.output is not None
            else input_path.with_name(f'{input_path.stem}_shuffled{input_path.suffix}')
        )

    rng = random.Random(args.seed)

    dataset = _load_json(input_path)
    if not isinstance(dataset, list):
        raise TypeError(f'Expected top-level JSON to be a list, got {type(dataset)}')

    stats = shuffle_qa_in_each_group(dataset, rng)
    _dump_json(output_path, dataset)

    seed_msg = str(args.seed) if args.seed is not None else 'None'
    print(
        'Done. Shuffled `qa` within each group. '\
        f'groups={stats["groups"]}, total_qa={stats["total_qa"]}, '\
        f'missing_qa_groups={stats["missing_qa_groups"]}, seed={seed_msg}.\n'\
        f'Output: {output_path.resolve()}'
    )


if __name__ == '__main__':
    main()
