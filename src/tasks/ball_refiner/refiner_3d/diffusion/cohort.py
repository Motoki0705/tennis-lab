"""Pin validation identities when changing only the training population."""
from __future__ import annotations

from typing import Any


def reference_validation(records: list[dict[str, Any]], reference: dict[str, Any]) -> set[str]:
    if reference['status'] != 'complete':
        raise ValueError('Need a complete reference training run')
    rows = [r for r in reference['read_rallies'] if r['rally_id'].startswith('val-')]
    expected = {r['rally_id']: r['npz_sha256'] for r in rows}
    if len(expected) != len(rows) or len(expected) != reference['config']['expected_counts']['val']:
        raise ValueError('Reference validation identities are incomplete/duplicated')
    actual = {r['rally_id']: r['npz_sha256'] for r in records if r['split'] == 'val'}
    if not expected or any(actual.get(key) != digest for key, digest in expected.items()):
        raise ValueError('Reference validation rally/hash mismatch')
    return set(expected)
