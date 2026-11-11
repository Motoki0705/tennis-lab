"""Run-specific fixed matrix and verified inputs for issue 1859."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.tasks.ball_detection.models.mdd.config import MDDHeatmapConfig
from src.tasks.ball_detection.models.mdd.multiscale_config import (
    MDDMultiScaleQueryConfig,
)
from src.tasks.ball_detection.models.mdd.stage_config import MDDStageHeatmapConfig
from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget

REPO = Path("/content/tennis-lab")
INPUTS = REPO / "outputs/ball_detection/arch-inputs"
ROOT = REPO / "outputs/ball_detection/train/arch_efficiency/s42-20261111"
REMOTE = "outputs/ball_detection/train/arch_efficiency/s42-20261111"
DRIVE = Path("/content/drive/MyDrive/tennis_lab")
CONFIG = Path("/tmp/ball-drive-mount.conf")
PREPARED = INPUTS / "capacity12k/prepared"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_config(case, *, levels=(8, 16, 32, 64), chunk=4):
    name = "common-dpt-T32" if case.startswith("C") else "query-d512-l6-T64"
    raw = dict(read(INPUTS / name / "run.json")["recipe"]["model"])
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        raw[key] = tuple(raw[key])
    if case == "C0":
        return MDDHeatmapConfig(**raw)
    if case.startswith("C"):
        kinds = {
            "C1": ("ib",) * 5,
            "C2": ("convnext", "convnext", "ffn", "ffn", "ffn"),
            "C3": ("convnext", "convnext", "ib", "ib", "ib"),
        }[case]
        return MDDStageHeatmapConfig(
            **raw,
            stage_blocks=kinds,
            expansion_ratios=((1.5,) * 5 if case == "C1" else (2.0,) * 5),
        )
    if case == "CA":
        from src.tasks.ball_detection.models.mdd.config import MDDMHCQueryConfig

        return MDDMHCQueryConfig(**raw)
    return MDDMultiScaleQueryConfig(
        **raw,
        spatial_layers=(
            ("M",) * 3 + ("C",) * 3 if case == "T1" else ("C",) * 3 + ("M",) * 3
        ),
        msda_levels=tuple(levels),
        msda_points=4,
        msda_chunk_size=chunk,
        msda_im2col_step=64,
    )


def budget(case, *, stage=0, smoke=False):
    if case.startswith("C"):
        raw = read(INPUTS / "common-dpt-T32/run.json")["recipe"]["budget"]
    else:
        source = read(INPUTS / "capacity12k/plan.json")
        raw = next(job for job in source["jobs"] if job["model"] == "d512-l12")[
            "stages"
        ][stage]["budget"]
    fields = {k: raw[k] for k in UpdateBudget.__dataclass_fields__}
    if smoke:
        fields["freeze_updates"] = 0
    return UpdateBudget(**fields)


def recipe(case):
    name = "common-dpt-T32" if case.startswith("C") else "query-d512-l6-T64"
    return read(INPUTS / name / "run.json")["recipe"]


def config_payload(case, levels):
    return asdict(model_config(case, levels=levels))
