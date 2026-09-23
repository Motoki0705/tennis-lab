"""Validate existing scenes and prepare task-owned features and video splits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dataset import load_clip_arrays
from src.tasks.slcs.data.dino_precompute import (
    FrameEncoder,
    PrecomputeReport,
    run_precompute,
)
from src.tasks.slcs.data.splits import (
    generate_overfit_splits,
    generate_video_splits,
    load_split_assignments,
    save_split_file,
)
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifestError,
)
from src.utils.checksum import dual_sha256
from src.utils.io import load_json

if TYPE_CHECKING:
    from src.tasks.slcs.configuration import SLCSGenerationConfig


@dataclass(frozen=True)
class GenerationReport:
    """Feature outcomes and split publication state."""

    features: PrecomputeReport
    split_reused: bool
    split_ready: bool

    @property
    def ok(self) -> bool:
        return self.features.ok and self.split_ready


def prepare_dataset(
    runtime: SLCSGenerationConfig,
    *,
    encoder_factory: Callable[[], FrameEncoder],
) -> GenerationReport:
    """Prepare one complete dataset, leaving every source scene unchanged.

    Validate every scene and any existing split before loading an encoder.
    Feature failures preserve completed camera annotations and prevent split
    publication. A later invocation validates and reuses completed features.
    """
    precompute = runtime.precompute
    data = precompute.data
    split = runtime.splits
    index = SLCSDataIndex.load(data.dataset_root)
    for ref in index.clips:
        load_clip_arrays(ClipManifest.load(index.clip_dir(ref)), config=data.pipeline)

    assignments = (
        generate_overfit_splits(index)
        if split.overfit
        else generate_video_splits(
            index, val_ratio=split.val_ratio, test_ratio=split.test_ratio, seed=split.seed
        )
    )
    val_ratio = 0.0 if split.overfit else split.val_ratio
    test_ratio = 0.0 if split.overfit else split.test_ratio
    split_reused = data.split_file.exists() and not split.overwrite
    if split_reused:
        saved = load_split_assignments(data.split_file, index)
        document = load_json(data.split_file)
        if saved != assignments or any(
            document.get(key) != value
            for key, value in (
                ("seed", split.seed),
                ("val_ratio", val_ratio),
                ("test_ratio", test_ratio),
            )
        ):
            raise DatasetManifestError(
                f"{data.split_file}: existing split does not match the requested "
                "dataset/settings. Set splits.overwrite=true to regenerate."
            )

    encoder: FrameEncoder | None = None

    def encode(frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        nonlocal encoder
        if encoder is None:
            encoder = encoder_factory()
        return encoder(frames)

    features = run_precompute(
        data.dataset_root,
        encode,
        data.pipeline.dino_spec,
        batch_size=precompute.batch_size,
        overwrite=precompute.overwrite,
        checkpoint_sha256=dual_sha256(precompute.checkpoint_path),
        generator={
            "script": "src.tasks.slcs.scripts.generate_dataset",
            "backbone": data.pipeline.dino_spec.backbone,
        },
    )
    if not features.ok:
        return GenerationReport(features, split_reused, split_ready=False)
    if not split_reused:
        save_split_file(
            data.split_file,
            assignments,
            seed=split.seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    return GenerationReport(features, split_reused, split_ready=True)
