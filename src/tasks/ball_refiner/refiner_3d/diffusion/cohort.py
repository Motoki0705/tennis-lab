"""Pin validation identities, including a reference that already uses a subset."""
from __future__ import annotations

from typing import Any


def reference_validation(records: list[dict[str, Any]], reference: dict[str, Any]) -> set[str]:
    if reference['status'] != 'complete':
        raise ValueError('Need a complete reference training run')
    rows = [r for r in reference['read_rallies'] if r['rally_id'].startswith('val-')]
    expected = {r['rally_id']: r['npz_sha256'] for r in rows}
    if len(expected) != len(rows):
        raise ValueError('Reference validation identities are incomplete/duplicated')
    actual = {r['rally_id']: r['npz_sha256'] for r in records if r['split'] == 'val'}
    reference_count = reference['config']['expected_counts']['val']
    if 'validation_reference' in reference:
        cohort = reference['validation_reference']
        if not isinstance(cohort, dict):
            raise ValueError('Reference validation subset must be explicit')
        selected, unused = cohort.get('rallies'), cohort.get('unused_val_rallies')
        if (not isinstance(selected, list) or not isinstance(unused, list)
                or any(not isinstance(key, str) or not key.startswith('val-') for key in selected + unused)):
            raise ValueError('Reference validation subset needs selected/unused identities')
        if (len(set(selected)) != len(selected) or len(set(unused)) != len(unused)
                or set(selected) != set(expected) or set(selected) & set(unused)
                or len(selected) + len(unused) != reference_count
                or not set(selected + unused) <= actual.keys()):
            raise ValueError('Reference validation subset partition is incomplete/duplicated')
    elif len(expected) != reference_count:
        raise ValueError('Reference validation identities are incomplete/duplicated')
    if not expected or any(actual.get(key) != digest for key, digest in expected.items()):
        raise ValueError('Reference validation rally/hash mismatch')
    return set(expected)
