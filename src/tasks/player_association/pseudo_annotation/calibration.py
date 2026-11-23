"""Blind visual calibration against human-reviewed Meiji 3cam boxes."""

from collections.abc import Iterable
from pathlib import Path

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.pseudo_annotation.images import prepare_images
from src.tasks.player_association.pseudo_annotation.job import (
    Candidate,
    Packet,
    frame_numbers,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.checksum import dual_sha256


def prepare_calibration(
    dataset: Path,
    clip_id: str,
    frames: Iterable[int],
    destination: Path,
) -> Packet:
    """Show only reviewed coordinates in geometric order, with human roles and IDs held out."""
    path = label_path(dataset, clip_id)
    clip = ClipManifest.load(path.parents[2])
    truth = ClipLabels.load(path)
    if (
        truth.clip_id != clip.clip_id
        or truth.num_frames != clip.num_frames
        or set(truth.cameras) != set(clip.camera_ids)
    ):
        raise ValueError("Human labels and source clip disagree")
    if (
        truth.annotated_frames is not None
        or sum(person.role == "player" for person in truth.people) != 2
    ):
        raise ValueError(
            "Calibration expects the existing human-reviewed singles reference"
        )
    indices = frame_numbers(frames, clip.num_frames)
    candidates = {
        frame: {
            camera: [
                Candidate(box=box)
                for box in sorted(rows.boxes_xyxy[rows.at(frame)].tolist())
            ]
            for camera, rows in truth.cameras.items()
        }
        for frame in indices
    }
    if any(not any(views.values()) for views in candidates.values()):
        raise ValueError("Calibration frames must contain reviewed boxes")
    return prepare_images(
        clip,
        candidates,
        dict.fromkeys(indices, ["human_calibration"]),
        destination,
        player_count=2,
        input_snapshot_sha256=dual_sha256(path),
        show_prior_association=False,
    )
