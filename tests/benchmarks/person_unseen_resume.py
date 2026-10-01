"""Authorize the run-17 preparation explicitly, without editing the person freeze."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from person_unseen_freeze import CODE  # type: ignore[import-not-found]

from src.tennis_scene.pipeline.definition import file_identity


def resumed_budget(frozen: dict[str, Any]) -> dict[str, Any]:
    """The 3-hour grant changes wall time and tightens output disk only."""
    return {**frozen['budget'], 'wall_seconds': 10770, 'outer_timeout_seconds': 10790,
            'disk_limit_bytes': 2_800_000_000}


def pushed_document(path: Path, commit: str, *, root: Path = CODE,
                    remote: str = 'origin/campaign930/i964-2-tracking') -> dict[str, Any]:
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=root, check=True)
    blob = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(root)}'], cwd=root)
    if path.read_bytes() != blob:
        raise ValueError('Execution addendum changed since the pushed commit')
    document: dict[str, Any] = json.loads(blob)
    return document


def load_addendum(path: Path, commit: str, freeze: Path, freeze_commit: str,
                  frozen: dict[str, Any], *, root: Path = CODE,
                  remote: str = 'origin/campaign930/i964-2-tracking') -> dict[str, Any]:
    document = pushed_document(path, commit, root=root, remote=remote)
    if document['schema'] != 'i964_unseen_resume_v1' or document['freeze'] != file_identity(freeze) \
            or document['freeze_commit'] != freeze_commit or document['report'] != frozen['report']:
        raise ValueError('Addendum must refer to the original freeze and output directory')
    if document['budget'] != resumed_budget(frozen) or document['allowed_missing_sides'] != ['video_001/clip_003']:
        raise ValueError('Addendum exceeds the run-17 execution-only authorization')
    if Path(document['previous_opening']['path']) != Path(frozen['report']) / 'opening.json':
        raise ValueError('Addendum must preserve the original opening receipt')
    for key in ('previous_opening', 'previous_stop'):
        record = document[key]
        if file_identity(Path(record['path'])) != record:
            raise ValueError(f'Original preparation receipt changed: {key}')
    opening = json.loads(Path(document['previous_opening']['path']).read_text())
    stop = json.loads(Path(document['previous_stop']['path']).read_text())
    for record in (opening, stop):
        if record['freeze_commit'] != freeze_commit or record['inference_attempts'] != 0 or record['scoring_batches'] != 0:
            raise ValueError('Only preparation with zero inference and scoring attempts may resume')
    if opening['freeze'] != document['freeze'] or stop['status'] != 'not_enqueued' \
            or stop['reserved_media_decoded'] or stop['reserved_person_labels_opened']:
        raise ValueError('Original stop must precede unseen inference and label opening')
    side = stop['side_reference']
    if file_identity(Path(side['path'])) != side:
        raise ValueError('Historical side evidence changed')
    return document


def load_preparation_addendum(path: Path, commit: str, execution: dict[str, Any],
                             report: Path, freeze_commit: str) -> dict[str, Any]:
    """The recorded path-contract failure may resume; inference still has no retry path."""
    document = pushed_document(path, commit)
    names = {'opening.json', 'resumed-opening-r17.json', 'preparation-stop-r17.json'}
    if document['schema'] != 'i964_unseen_preparation_resume_v1' or document['execution_addendum'] != execution \
            or document['freeze_commit'] != freeze_commit or set(document['previous_files']) != names:
        raise ValueError('Preparation addendum must retain the original execution authorization and receipts')
    for name, identity in document['previous_files'].items():
        if file_identity(report / name) != identity:
            raise ValueError(f'Preparation receipt changed: {name}')
    stop = json.loads((report / 'preparation-stop-r17.json').read_text())
    if stop['failure']['exception_type'] != 'PathContractError' or stop['inference_attempts'] != 0 \
            or stop['scoring_batches'] != 0 or stop['reserved_person_labels_opened'] \
            or stop['reserved_media_decoded'] or stop['queue_job'] is not None:
        raise ValueError('Only the documented pre-inference path-contract failure may resume')
    return document


def require_unstarted_directory(report: Path, preparation: dict[str, Any] | None = None) -> None:
    """Only the original opening may exist; neither preparation nor inference silently retries."""
    allowed = {'opening.json'} if preparation is None else set(preparation['previous_files'])
    if {p.name for p in report.iterdir()} != allowed:
        raise FileExistsError('Resumed preparation requires only the original opening receipt')
