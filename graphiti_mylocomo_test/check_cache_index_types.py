"""Quick sanity checks for semantic cache index types."""

from __future__ import annotations

import sys
from typing import Iterable

import numpy as np

from graphiti_core.cache.vector_index import FAISS_AVAILABLE, create_vector_index


INDEX_TYPES: list[str] = ['numpy', 'faiss_exact', 'faiss_hnsw']


def _make_vectors(count: int, dim: int, seed: int = 7) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, dim)).astype(np.float32).tolist()


def _check_index(index_type: str, dim: int = 8) -> tuple[bool, str]:
    if index_type != 'numpy' and not FAISS_AVAILABLE:
        return False, f'{index_type}: FAIL (FAISS not available)'

    index = create_vector_index(
        dim,
        index_type=index_type,
        hnsw_m=16,
        hnsw_ef_search=32,
        hnsw_ef_construction=80,
    )

    vectors = _make_vectors(4, dim)
    for vec in vectors:
        index.add(vec)

    results = index.search(vectors[0], k=3)
    if not results:
        return False, f'{index_type}: FAIL (no results)'

    if results[0][1] < results[-1][1]:
        return False, f'{index_type}: FAIL (results not sorted)'

    return True, f'{index_type}: OK (top_k={len(results)})'


def main(index_types: Iterable[str] = INDEX_TYPES) -> int:
    failed = 0
    for index_type in index_types:
        ok, message = _check_index(index_type)
        print(message)
        if not ok:
            failed += 1

    if failed:
        print(f'FAILED: {failed} index type(s)')
        return 1

    print('ALL OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
