"""Benchmark domains, deterministic sampling, and ground-truth loading.

Both domains reuse the production input layer rather than re-deriving the
synthetic V3 semantics or the TennisCourtDetector annotation contract, so the
benchmark cannot drift from what a model was trained and validated on.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.configuration import (
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtRawSample,
    CourtSampleRecord,
    CourtSourceSplit,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.tasks.court_detection.evaluation.contracts import (
    KEYPOINT_COUNT,
    DomainName,
    LoadedSample,
    SampleRef,
)
from src.tasks.court_detection.evaluation.settings import (
    BenchmarkDomainSettings,
    BenchmarkSelectionSettings,
)
from src.tasks.court_detection.geometry.homography import court_template_xy


@dataclass(frozen=True, slots=True)
class DomainRecords:
    """One benchmark domain bound to its validated source records."""

    domain: DomainName
    display_name: str
    split: str
    input_layer: CourtInput
    records: Mapping[str, CourtSampleRecord]

    def record(self, sample_id: str) -> CourtSampleRecord:
        try:
            return self.records[sample_id]
        except KeyError as error:
            raise KeyError(
                f"Benchmark sample {sample_id!r} is absent from domain {self.domain!r}."
            ) from error


def build_domain_records(
    settings: BenchmarkDomainSettings,
    *,
    derived_target_root: Path,
) -> DomainRecords:
    """Resolve every accepted record of one domain through the input layer."""
    store = CourtDerivedTargetStore(derived_target_root)
    input_layer = build_court_input(
        settings.source,
        target_store=store,
        requested_splits=_requested_splits(settings),
    )
    records = input_layer.records(cast("CourtSourceSplit", settings.split))
    by_id: dict[str, CourtSampleRecord] = {}
    for record in records:
        if record.sample_id in by_id:
            raise ValueError(
                f"Domain {settings.domain!r} contains a duplicate sample id."
            )
        by_id[record.sample_id] = record
    if not by_id:
        raise ValueError(f"Domain {settings.domain!r} contains no accepted samples.")
    return DomainRecords(
        domain=settings.domain,
        display_name=settings.display_name,
        split=settings.split,
        input_layer=input_layer,
        records=by_id,
    )


def _requested_splits(
    settings: BenchmarkDomainSettings,
) -> tuple[CourtSourceSplit, ...] | None:
    """Return the split subset the benchmark evaluates, when it is available.

    The real domain maps ``train -> train`` like production does, but the
    benchmark measures the validation split only.  Asking the input layer for
    that split keeps the run from preflighting thousands of unrelated train
    images.  The synthetic input validates every configured split by contract,
    so it keeps the full read.
    """
    if isinstance(settings.source, TennisCourtDetectorSourceConfig):
        return (cast("CourtSourceSplit", settings.split),)
    return None


def sample_ref(domain: DomainRecords, record: CourtSampleRecord) -> SampleRef:
    """Convert one validated record into its manifest identity."""
    payload = record.payload
    scene = payload.get("scene_id")
    if scene is None:
        scene = "tennis_court_detector"
    width = payload.get("width")
    height = payload.get("height")
    digest = payload.get("source_target_sha256")
    if type(width) is not int or type(height) is not int:
        raise ValueError("Court records must carry an integer source resolution.")
    if type(digest) is not str:
        raise ValueError("Court records must carry a source target digest.")
    group = payload.get("trajectory_group_id")
    return SampleRef(
        domain=domain.domain,
        sample_id=record.sample_id,
        scene_id=str(scene),
        trajectory_group_id=None if group is None else str(group),
        split=record.split,
        source_target_sha256=digest,
        width=width,
        height=height,
    )


def select_manifest_refs(
    refs: Sequence[SampleRef],
    *,
    selection: BenchmarkSelectionSettings,
) -> tuple[SampleRef, ...]:
    """Deterministically pick samples without favouring one scene or group.

    Groups (scene, trajectory group) are visited round-robin so a truncated run
    still spreads its samples across the whole test set.  The seed chooses the
    group visit order and rotates each group's frames, so a cap never keeps only
    the first frames of every trajectory.  The same seed always yields the same
    manifest.
    """
    ordered = sorted(refs, key=lambda item: item.sample_id)
    maximum = selection.max_samples_per_domain
    if maximum is None or maximum >= len(ordered):
        return tuple(ordered)
    buckets: dict[tuple[str, str], list[SampleRef]] = {}
    for item in ordered:
        buckets.setdefault(item.group_key, []).append(item)
    keys = sorted(buckets)
    generator = random.Random(selection.seed)
    generator.shuffle(keys)
    rotated: dict[tuple[str, str], list[SampleRef]] = {}
    for key in keys:
        bucket = buckets[key]
        offset = generator.randrange(len(bucket))
        rotated[key] = bucket[offset:] + bucket[:offset]
    selected: list[SampleRef] = []
    offset = 0
    while len(selected) < maximum:
        progressed = False
        for key in keys:
            bucket = rotated[key]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            progressed = True
            if len(selected) >= maximum:
                break
        if not progressed:
            break
        offset += 1
    return tuple(sorted(selected, key=lambda item: item.sample_id))


def load_sample(domain: DomainRecords, ref: SampleRef) -> LoadedSample:
    """Load RGB and the authoritative ground truth for one manifest entry."""
    record = domain.record(ref.sample_id)
    raw = domain.input_layer.load(record)
    image = _rgb_array(raw)
    if (image.shape[1], image.shape[0]) != (ref.width, ref.height):
        raise ValueError(
            "Loaded RGB resolution disagrees with the manifest entry; the source "
            "dataset changed after the manifest was written."
        )
    channels = raw.keypoint_channels
    if channels is None:
        raise ValueError("Court input layer produced no keypoint channels.")
    if channels.points_xy.shape != (KEYPOINT_COUNT, 1, 2):
        raise ValueError(
            "The benchmark requires one target-court keypoint per channel, got "
            f"{tuple(channels.points_xy.shape)}."
        )
    points = channels.points_xy[:, 0, :].detach().cpu().numpy().astype(np.float64)
    visible = channels.point_visible[:, 0].detach().cpu().numpy().astype(bool)
    physical = channels.physical_indices[:, 0].detach().cpu().numpy()
    if sorted(physical.tolist()) != list(range(KEYPOINT_COUNT)):
        raise ValueError(
            "Court keypoint channels must cover the canonical 14-point permutation."
        )
    template = court_template_xy(KEYPOINT_COUNT)[physical].astype(np.float64)
    strata = _strata(domain, raw, visible)
    return LoadedSample(
        ref=ref,
        image_rgb=image,
        template_xy=template,
        gt_keypoints_xy=np.where(visible[:, None], points, np.nan),
        gt_visible=visible,
        strata=strata,
    )


def _rgb_array(raw: CourtRawSample) -> NDArray[np.uint8]:
    image = raw.image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Court RGB images must decode to H x W x 3 uint8.")
    return array


def _strata(
    domain: DomainRecords,
    raw: CourtRawSample,
    visible: NDArray[np.bool_],
) -> dict[str, str]:
    """Return the ground-truth strata axes this domain can actually support."""
    strata = {"visible_kp_count": str(int(visible.sum()))}
    if domain.domain == "real_validation":
        strata["scene"] = "tennis_court_detector"
        return strata
    scene_id = raw.metadata.scene_id
    if scene_id is None:
        raise ValueError("Synthetic Court samples must carry a scene id.")
    strata["scene"] = scene_id
    strata["coverage_mode"] = _target_court_coverage_mode(
        domain.record(raw.sample_id),
        source_sample_id=raw.metadata.source_sample_id,
    )
    return strata


def _target_court_coverage_mode(
    record: CourtSampleRecord, *, source_sample_id: str
) -> str:
    """Read the target court's published coverage mode from its labels."""
    target_court_id = record.payload.get("target_court_id")
    if type(target_court_id) is not str or not target_court_id:
        raise ValueError("Synthetic Court records must name their target court.")
    parsed = json.loads(record.annotation_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping) or parsed.get("sample_id") != source_sample_id:
        raise ValueError(
            "Synthetic Court labels disagree with the manifest sample identity."
        )
    projection = parsed.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Synthetic Court labels must publish a projection mapping.")
    courts = projection.get("courts")
    if not isinstance(courts, list):
        raise ValueError("Synthetic Court projection must list its courts.")
    matches = [
        court.get("coverage_mode")
        for court in courts
        if isinstance(court, Mapping)
        and court.get("court_instance_id") == target_court_id
    ]
    if len(matches) != 1 or type(matches[0]) is not str:
        raise ValueError(
            "Synthetic Court target court must publish exactly one coverage mode."
        )
    return str(matches[0])


__all__ = [
    "DomainRecords",
    "build_domain_records",
    "load_sample",
    "sample_ref",
    "select_manifest_refs",
]
