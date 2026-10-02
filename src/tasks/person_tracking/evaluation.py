"""Predeclared partial-reference camera-local IDF1, distinct from #933 metrics.

Reference duplicates of one person/frame form one unit. All labelled players
stay in the IDFN denominator. Selected known nonplayers contribute IDFP, while
unlabelled/ambiguous predictions are separately counted, never presumed false.
IDs remain raw tracker IDs after selection, not court-linked group IDs.
"""

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels
from src.utils.geometry.bbox import pairwise_iou


def tracking_units(tracks: CameraTracks, selected: np.ndarray, labels: ClipLabels) -> tuple[list[dict[str, Any]], int]:
    if selected.dtype != np.bool_ or selected.shape != tracks.observed.shape \
            or selected.shape[1] != labels.num_frames or (selected & ~tracks.observed).any():
        raise ValueError('Selection must contain only real observations on the labelled timeline')
    camera = tracks.camera.camera_id
    reference = labels.cameras[camera]
    units = []
    unlabelled = 0
    for frame in range(labels.num_frames):
        at = reference.at(frame)
        people = reference.person_index[at]
        boxes = reference.boxes_xyxy[at]
        identities = np.unique(people[people != AMBIGUOUS])
        predicted = sorted(np.flatnonzero(selected[:, frame]), key=lambda r: tracks.track_ids[r])
        overlap = pairwise_iou(tracks.boxes_xyxy[predicted, frame], boxes)
        iou = np.column_stack([overlap[:, people == p].max(1) for p in identities]) \
            if len(identities) else np.empty((len(predicted), 0))
        admissible = iou >= .5
        # One extra valid match must outweigh every possible IoU-sum gain.
        reward = np.where(admissible, min(iou.shape) + 1. + iou, 0.)
        pi, gi = linear_sum_assignment(-reward)
        matches = {int(g): int(tracks.track_ids[predicted[p]]) for p, g in zip(pi, gi, strict=True) if admissible[p, g]}
        unlabelled += len(predicted) - len(matches)
        player_bottoms = {}
        for g, p in enumerate(identities):
            if labels.people[p].role == 'player':
                candidates = boxes[people == p]
                biggest = int(np.prod(candidates[:, 2:] - candidates[:, :2], axis=1).argmax())
                player_bottoms[g] = float(candidates[biggest, 3])
        two = len(player_bottoms) == 2 and len(set(player_bottoms.values())) == 2
        for g, p in enumerate(identities):
            stratum = 'unknown'
            if two and g in player_bottoms:
                stratum = 'near' if player_bottoms[g] == max(player_bottoms.values()) else 'far'
            units.append({'clip': labels.clip_id, 'camera': camera, 'frame': frame,
                          'person': labels.people[p].person_id, 'role': labels.people[p].role,
                          'near_far': stratum, 'track_id': matches.get(g)})
    return units, unlabelled


def score_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool counts after independent identity assignment per clip and camera."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[unit['clip'], unit['camera']].append(unit)
    idtp = 0
    mapping: dict[tuple[str, str, int], str] = {}
    for (clip, camera), rows in sorted(groups.items()):
        people = sorted({r['person'] for r in rows if r['role'] == 'player'})
        ids = sorted({r['track_id'] for r in rows if r['track_id'] is not None})
        counts = Counter((r['person'], r['track_id']) for r in rows if r['role'] == 'player' and r['track_id'] is not None)
        matrix = np.asarray([[counts[p, i] for i in ids] for p in people], np.int64).reshape(len(people), len(ids))
        pi, ti = linear_sum_assignment(-matrix)
        idtp += int(matrix[pi, ti].sum())
        mapping.update({(clip, camera, ids[t]): people[p] for p, t in zip(pi, ti, strict=True) if matrix[p, t] > 0})
    players = [u for u in units if u['role'] == 'player']
    kept = sum(u['track_id'] is not None for u in players)
    nonplayers = [u for u in units if u['role'] == 'non_player']
    retained = sum(u['track_id'] is not None for u in nonplayers)
    idfn, idfp = len(players) - idtp, kept + retained - idtp
    trajectories: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    individuals: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in players:
        trajectories[row['clip'], row['camera'], row['person']].append(row)
        individuals[row['clip'], row['person']].append(row)
    switches, fragments = [], []
    for rows in trajectories.values():
        previous_id = None
        previous_frame = -2
        ever_matched = missed = False
        for row in sorted(rows, key=lambda r: r['frame']):
            if row['frame'] != previous_frame + 1:
                ever_matched = missed = False  # unknown reference gap, not a tracker miss
            track = row['track_id']
            if track is not None:
                if previous_id is not None and track != previous_id:
                    switches.append({**row, 'previous_track_id': previous_id})
                if ever_matched and missed:
                    fragments.append(dict(row))
                previous_id = track
                ever_matched, missed = True, False
            elif ever_matched:
                missed = True
            previous_frame = row['frame']
    denominator = 2 * idtp + idfp + idfn
    return {'idtp': idtp, 'idfp': idfp, 'idfn': idfn, 'idf1': 2 * idtp / denominator if denominator else None,
            'player_units': len(players), 'player_units_kept': kept,
            'player_identities': len(individuals),
            'player_identities_kept50': sum(sum(u['track_id'] is not None for u in rows) >= len(rows) / 2 for rows in individuals.values()),
            'nonplayer_units': len(nonplayers), 'nonplayer_units_kept': retained,
            'id_switches': len(switches), 'fragments': len(fragments),
            'switch_events': switches, 'fragment_events': fragments,
            'mapping': [{'clip': c, 'camera': cam, 'track_id': t, 'person': p} for (c, cam, t), p in sorted(mapping.items())]}


def stratified_scores(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for camera in ('all', 'cam0', 'cam1', 'cam2'):
        for near_far in ('all', 'near', 'far', 'unknown'):
            subset = [u for u in units if (camera == 'all' or u['camera'] == camera)
                      and (near_far == 'all' or u['near_far'] == near_far)]
            metrics = score_units(subset)
            rows.append({'camera': camera, 'near_far': near_far,
                         **{k: v for k, v in metrics.items() if k not in ('mapping', 'switch_events', 'fragment_events')}})
    return rows
