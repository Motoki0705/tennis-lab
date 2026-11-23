"""A pinned annotation packet, its response contract and v1 label ingest."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.tasks.player_association.evaluation.labels import (
    AMBIGUOUS,
    PSEUDO_PROVENANCE,
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.tennis_scene.generate_dataset.manifest import (
    split_clip_id,
    validate_id_component,
)
from src.utils.checksum import dual_sha256

MODEL = "gpt-6.1-terra"
EFFORT = "xhigh"
NON_PLAYER = "non_player"
UNCERTAIN = "ambiguous"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_object)


def frame_numbers(frames: Iterable[int], total: int) -> list[int]:
    result = list(frames)
    if not result or any(
        type(frame) is not int or not 0 <= frame < total for frame in result
    ):
        raise ValueError("Supply a nonempty set of valid zero-based source frames")
    if result != sorted(set(result)):
        raise ValueError("Source frames must be sorted and unique")
    return result


class Candidate(StrictRecord):
    box: list[float]
    raw_track: int | None = None
    prior_player: int | None = None  # zero-based ID of the old pipeline

    def model_post_init(self, context: Any, /) -> None:
        if (
            len(self.box) != 4
            or self.box[2] <= self.box[0]
            or self.box[3] <= self.box[1]
        ):
            raise ValueError("Candidate must be a positive xyxy box")


class Panel(StrictRecord):
    camera: str
    candidates: list[Candidate]


class Frame(StrictRecord):
    index: int
    panels: list[Panel]
    strata: list[str]
    image_sha256: Sha256

    def model_post_init(self, context: Any, /) -> None:
        if (
            not self.strata
            or len(self.strata) != len(set(self.strata))
            or any(not name.strip() for name in self.strata)
        ):
            raise ValueError(
                f"Frame {self.index} must identify its unique sampling strata"
            )

    @property
    def image_name(self) -> str:
        return f"frame_{self.index:06d}.jpg"


class Packet(StrictRecord):
    schema_version: Literal[1]
    dataset_id: str
    clip_id: str
    num_frames: int
    start: int
    stop: int
    camera_ids: list[str]
    player_count: Literal[2, 4]
    show_prior_association: bool
    input_snapshot_sha256: Sha256
    clip_manifest_sha256: Sha256
    media_sha256: dict[str, Sha256]
    frames: list[Frame]

    def model_post_init(self, context: Any, /) -> None:
        validate_id_component(self.dataset_id, field_name="dataset_id")
        split_clip_id(self.clip_id)
        indices = frame_numbers((frame.index for frame in self.frames), self.num_frames)
        if (
            not 0
            <= self.start
            <= indices[0]
            <= indices[-1]
            < self.stop
            <= self.num_frames
        ):
            raise ValueError(
                "Annotated frames must lie within one half-open clip interval"
            )
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("Packet camera IDs must be nonempty and unique")
        for camera in self.camera_ids:
            validate_id_component(camera, field_name="camera_id")
        if set(self.media_sha256) != set(self.camera_ids):
            raise ValueError("Media digests must cover all cameras")
        for frame in self.frames:
            if [panel.camera for panel in frame.panels] != self.camera_ids:
                raise ValueError(
                    f"Frame {frame.index} does not have all cameras in packet order"
                )
            if not self.show_prior_association and any(
                box.raw_track is not None or box.prior_player is not None
                for panel in frame.panels
                for box in panel.candidates
            ):
                raise ValueError("Blind packet contains tracker or association hints")

    @property
    def player_ids(self) -> list[str]:
        return [f"P{i}" for i in range(1, self.player_count + 1)]

    @classmethod
    def load(cls, path: Path, *, verify_images: bool = True) -> Packet:
        packet: Packet = cls.model_validate(read_json(path / "packet.json"))
        if verify_images:
            for frame in packet.frames:
                if dual_sha256(path / frame.image_name) != frame.image_sha256:
                    raise ValueError(f"Image checksum mismatch: {frame.image_name}")
        return packet


class PanelLabels(StrictRecord):
    camera: str
    labels: list[str]


class FrameLabels(StrictRecord):
    frame: int
    panels: list[PanelLabels]


class Response(StrictRecord):
    input_sha256: str
    player_descriptions: list[str]
    frames: list[FrameLabels]


def ingest(
    packet: Packet, response: Response, *, input_sha256: str, prompt_sha256: str
) -> ClipLabels:
    """Use only pinned coordinates. Reject an unaccounted-for frame, view or box."""
    if response.input_sha256 != input_sha256:
        raise ValueError("Response input SHA-256 does not match its packet")
    if len(response.player_descriptions) != packet.player_count or any(
        not description.strip() for description in response.player_descriptions
    ):
        raise ValueError("Response must describe every player slot, in P1 to Pn order")
    if [frame.frame for frame in response.frames] != [
        frame.index for frame in packet.frames
    ]:
        raise ValueError(
            "Response frames must cover the requested frames once and in packet order"
        )

    names = [*packet.player_ids, NON_PLAYER]
    people = tuple(
        LabelledPerson(name, "player", description)
        for name, description in zip(
            packet.player_ids, response.player_descriptions, strict=True
        )
    )
    people += (
        LabelledPerson(NON_PLAYER, NON_PLAYER, "Outside the players in this match"),
    )
    codes = {name: index for index, name in enumerate(names)}
    codes[UNCERTAIN] = AMBIGUOUS
    rows: dict[str, list[tuple[int, int, list[float]]]] = {
        camera: [] for camera in packet.camera_ids
    }
    for target, annotated in zip(packet.frames, response.frames, strict=True):
        if [panel.camera for panel in annotated.panels] != packet.camera_ids:
            raise ValueError(
                f"Frame {target.index} must cover each camera once and in packet order"
            )
        for panel, answer in zip(target.panels, annotated.panels, strict=True):
            if len(panel.candidates) != len(answer.labels):
                raise ValueError(
                    f"Frame {target.index} {panel.camera}: expected {len(panel.candidates)} box labels"
                )
            if unknown := set(answer.labels) - codes.keys():
                raise ValueError(
                    f"Frame {target.index} {panel.camera}: unknown labels {sorted(unknown)}"
                )
            rows[panel.camera].extend(
                (target.index, codes[name], candidate.box)
                for name, candidate in zip(answer.labels, panel.candidates, strict=True)
            )

    labels = ClipLabels(
        packet.clip_id,
        packet.num_frames,
        people,
        {
            camera: CameraLabels(
                np.asarray([frame for frame, _, _ in values], dtype=np.int64),
                np.asarray([person for _, person, _ in values], dtype=np.int64),
                np.asarray([box for _, _, box in values], dtype=np.float64).reshape(
                    -1, 4
                ),
            )
            for camera, values in rows.items()
        },
        {
            PSEUDO_PROVENANCE: {
                "schema_version": 1,
                "dataset_id": packet.dataset_id,
                "annotated_frames": [frame.index for frame in packet.frames],
                "frame_strata": {
                    str(frame.index): frame.strata for frame in packet.frames
                },
                "interval": [packet.start, packet.stop],
                "model": MODEL,
                "effort": EFFORT,
                "input_sha256": input_sha256,
                "prompt_sha256": prompt_sha256,
                "input_snapshot_sha256": packet.input_snapshot_sha256,
                "clip_manifest_sha256": packet.clip_manifest_sha256,
                "media_sha256": packet.media_sha256,
                "show_prior_association": packet.show_prior_association,
            }
        },
    )
    # v1 writes coordinates rounded to 0.1 px. Fail before publishing if rounding
    # makes a candidate collapse or otherwise invalidates the reader contract.
    return ClipLabels.from_json(labels.to_json())
