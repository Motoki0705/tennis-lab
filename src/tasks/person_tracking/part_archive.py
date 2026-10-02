"""Read the recorded native KPR feature contract without a cosine proxy."""
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.player_association.appearance.kpr import WEIGHT_SHA256
from src.tasks.player_association.appearance.parts import NativeParts


def load_part_features(path: Path) -> tuple[list[DetectionFeatures], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'rows', 'boxes', 'scores', 'poses', 'offsets', 'part_embeddings', 'part_visibility', 'metadata'}:
            raise ValueError('Native feature fields differ from the archive contract')
        arrays = {key: saved[key] for key in saved.files}
    meta = json.loads(str(arrays['metadata']))
    if meta['schema'] != 'person_kpr_native_features_v1' or meta['weight_sha256'] != WEIGHT_SHA256:
        raise ValueError('Native archive schema/weight identity differs')
    offsets, rows = arrays['offsets'], arrays['rows']
    if rows.dtype != np.int64 or not np.array_equal(rows, np.arange(len(rows))) \
            or offsets.ndim != 1 or offsets.dtype != np.int64 or len(offsets) < 2 or offsets[0] != 0 \
            or offsets[-1] != len(rows) or (np.diff(offsets) < 0).any() \
            or arrays['part_embeddings'].shape != (len(rows), 6, 512):
        raise ValueError('Native archive rows/offsets/part shape differ')
    result = []
    for frame, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        parts = NativeParts(arrays['part_embeddings'][start:end], arrays['part_visibility'][start:end])
        result.append(DetectionFeatures(frame, rows[start:end], arrays['boxes'][start:end], arrays['scores'][start:end],
                                       arrays['poses'][start:end], np.zeros((end - start, 1), np.float32),
                                       np.zeros(end - start, bool), parts))
    return result, meta
