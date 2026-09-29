"""One-box group timelines for the existing v3 cross-camera safety checks.

The selection mask keeps all real observations, including handoff duplicates.
Only this association adapter reduces a <=0.2s handoff to one box per group
and frame, preferring the earlier raw track ID (never score or a label).
The origin row map makes that reduction auditable and maps IDs back exactly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import CameraTracks


def linked_timeline(tracks: CameraTracks, diagnostic: dict[str, Any]) -> tuple[CameraTracks, np.ndarray]:
    groups = [g for g in diagnostic['groups'] if g['selected']]
    frames = tracks.observed.shape[1]
    origins: np.ndarray = np.full((len(groups), frames), -1, np.int64)
    boxes = np.zeros((len(groups), frames, 4), np.float32)
    appearances = []
    for index, group in enumerate(groups):
        fragments = [diagnostic['fragments'][f] for f in group['fragments']]
        for fragment in sorted(fragments, key=lambda f: (f['track_id'], f['start'])):
            row = fragment['row']
            at = np.arange(fragment['start'], fragment['end'])
            at = at[tracks.observed[row, at] & (origins[index, at] < 0)]
            origins[index, at] = row
            boxes[index, at] = tracks.boxes_xyxy[row, at]
        feature_frames, feature_rows = [], []
        if tracks.appearance is not None:
            for row in np.unique(origins[index][origins[index] >= 0]):
                value = tracks.appearance[row]
                keep = origins[index, value.frames] == row
                if keep.any():
                    feature_frames.append(value.frames[keep])
                    feature_rows.append(value.embeddings[keep])
        if feature_frames:
            fs, es = np.concatenate(feature_frames), np.concatenate(feature_rows)
            order = np.argsort(fs, kind='stable')
            appearances.append(TrackAppearance(fs[order], es[order]))
        else:
            appearances.append(TrackAppearance(np.empty(0, np.int64), np.empty((0, 0), np.float32)))
    return CameraTracks(tracks.camera, tracks.image_size, np.arange(len(groups), dtype=np.int64),
        boxes, origins >= 0, appearances), origins
