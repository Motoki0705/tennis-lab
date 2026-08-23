"""Deterministic three-phase manifest generation for Court query ablations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

from src.tasks.court_detection.experiments.configuration import (
    DecoderFamily,
    DecoderSize,
    QueryAblationConfig,
    SupervisionName,
)

JsonValue: TypeAlias = Any

MANIFEST_SCHEMA = "court_query_ablation_manifest_v1"
PHASE_ORDER = ("encoder_first", "decoder_second", "supervision_third")
SELECTED_ENCODER_PLACEHOLDER = "__SELECTED_ENCODER_DEPTH__"
SELECTED_DECODER_FAMILY_PLACEHOLDER = "__SELECTED_DECODER_FAMILY__"
SELECTED_DECODER_SIZE_PLACEHOLDER = "__SELECTED_DECODER_SIZE__"


def build_ablation_manifest(config: QueryAblationConfig) -> dict[str, JsonValue]:
    """Build all runs in strict encoder, decoder, supervision order."""
    runs: list[dict[str, JsonValue]] = []
    for depth in config.encoder_first.depths:
        for seed in config.seeds:
            runs.append(_encoder_run(config, depth=depth, seed=seed))
    for family in config.decoder_second.families:
        for size in config.decoder_second.sizes:
            for seed in config.seeds:
                runs.append(_decoder_run(config, family=family, size=size, seed=seed))
    for supervision in config.supervision_third.variants:
        for seed in config.seeds:
            runs.append(_supervision_run(config, supervision=supervision, seed=seed))
    manifest: dict[str, JsonValue] = {
        "schema": MANIFEST_SCHEMA,
        "phase_order": list(PHASE_ORDER),
        "fixed_contract": {
            "seeds": list(config.seeds),
            "epochs": config.epochs,
            "input_hw": [config.image_height, config.image_width],
            "resize": "isotropic_fit_letterbox",
            "preserve_fx_fy": config.preserve_fx_fy,
            "hflip": config.hflip,
            "affine": config.affine,
            "shear": config.shear,
            "perspective": config.perspective,
            "training_entrypoint": config.train_module,
            "python_executable": config.python_executable,
            "gpu_execution_policy": "enqueue_with_repository_training_queue",
        },
        "selection_rules": {
            "encoder": {
                "reference_depth": config.encoder_first.reference_depth,
                "tolerance_ratio": config.encoder_first.tolerance_ratio,
                "metrics": [
                    "kp_mean_distance_px",
                    "pose_translation_l2_m",
                    "pose_rotation_geodesic_deg",
                    "pose_focal_relative_error",
                ],
                "rule": (
                    "smallest depth whose three-seed means are all within 5% of "
                    "depth-8; use depth-8 if none"
                ),
            },
            "decoder": {
                "reference": {"family": "dpt", "size": "base"},
                "tolerance_ratio": config.decoder_second.tolerance_ratio,
                "metric": "kp_mean_distance_px",
                "rule": (
                    "minimum decoder MACs, then decoder parameters, among candidates "
                    "within 5% KP mean-distance degradation of DPT-base"
                ),
            },
        },
        "selected": {
            "encoder_depth": config.selected_encoder_depth,
            "decoder_family": config.selected_decoder_family,
            "decoder_size": config.selected_decoder_size,
        },
        "runs": runs,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    validate_ablation_manifest(manifest, require_resolved=False)
    return manifest


def _base_argv(config: QueryAblationConfig, *, run_id: str, seed: int) -> list[str]:
    composition = config.composition
    return [
        config.python_executable,
        "-m",
        config.train_module,
        f"data/source={composition.source}",
        f"data.source.keypoint_court_scope={composition.keypoint_court_scope}",
        f"data/augmentation={composition.augmentation}",
        f"model={composition.model}",
        "model.preset=raw",
        "model/task_encoder=query_base",
        f"model/heads={composition.heads}",
        f"training.trainer.max_epochs={config.epochs}",
        f"run.seed={seed}",
        f"run.output_dir=court_detection/query_ablation/{run_id}",
    ]


def _encoder_run(
    config: QueryAblationConfig,
    *,
    depth: int,
    seed: int,
) -> dict[str, JsonValue]:
    final_tap = depth - 1
    run_id = f"encoder-depth-{depth:02d}-seed-{seed}"
    argv = _base_argv(config, run_id=run_id, seed=seed)
    argv.extend(
        [
            f"data/processing={config.composition.processing_kp}",
            f"loss={config.composition.loss_pose}",
            "model/decoder=query_linear_base",
            f"model.task_encoder.depth={depth}",
            f"model.task_encoder.tap_indices=[{final_tap}]",
            f"model.decoder.tap_indices=[{final_tap}]",
        ]
    )
    return _run_record(
        run_id=run_id,
        phase="encoder_first",
        phase_order=1,
        seed=seed,
        architecture={
            "encoder_depth": depth,
            "hidden_dim": config.encoder_first.hidden_dim,
            "num_heads": config.encoder_first.num_heads,
            "decoder_family": "linear",
            "decoder_size": "base",
            "encoder_taps": [final_tap],
        },
        supervision="kp+pose",
        argv=argv,
        unresolved=[],
    )


def _decoder_run(
    config: QueryAblationConfig,
    *,
    family: DecoderFamily,
    size: DecoderSize,
    seed: int,
) -> dict[str, JsonValue]:
    depth = config.selected_encoder_depth
    depth_token = str(depth) if depth is not None else SELECTED_ENCODER_PLACEHOLDER
    run_id = f"decoder-{family}-{size}-seed-{seed}"
    unresolved = [] if depth is not None else ["selected_encoder_depth"]
    argv = _base_argv(config, run_id=run_id, seed=seed)
    argv.extend(
        [
            f"data/processing={config.composition.processing_kp}",
            f"loss={config.composition.loss_pose}",
            f"model/decoder=query_{family}_{size}",
            f"model.task_encoder.depth={depth_token}",
        ]
    )
    taps: list[int] | str
    if depth is None:
        taps = SELECTED_ENCODER_PLACEHOLDER
    else:
        taps = _decoder_taps(depth, family=family, size=size)
        rendered = _render_int_list(taps)
        argv.extend(
            [
                f"model.task_encoder.tap_indices={rendered}",
                f"model.decoder.tap_indices={rendered}",
            ]
        )
        if family == "dpt":
            factors = _dpt_factors(len(taps))
            argv.extend(
                [
                    f"model.decoder.fusion_levels={len(taps)}",
                    f"model.decoder.reassemble_factors={_render_float_list(factors)}",
                ]
            )
    return _run_record(
        run_id=run_id,
        phase="decoder_second",
        phase_order=2,
        seed=seed,
        architecture={
            "encoder_depth": depth if depth is not None else depth_token,
            "hidden_dim": config.encoder_first.hidden_dim,
            "num_heads": config.encoder_first.num_heads,
            "decoder_family": family,
            "decoder_size": size,
            "encoder_taps": taps,
        },
        supervision="kp+pose",
        argv=argv,
        unresolved=unresolved,
    )


def _supervision_run(
    config: QueryAblationConfig,
    *,
    supervision: SupervisionName,
    seed: int,
) -> dict[str, JsonValue]:
    depth = config.selected_encoder_depth
    family = config.selected_decoder_family
    size = config.selected_decoder_size
    depth_token = str(depth) if depth is not None else SELECTED_ENCODER_PLACEHOLDER
    family_token = family if family is not None else SELECTED_DECODER_FAMILY_PLACEHOLDER
    size_token = size if size is not None else SELECTED_DECODER_SIZE_PLACEHOLDER
    run_id = f"supervision-{supervision.replace('+', '_')}-seed-{seed}"
    unresolved: list[str] = []
    if depth is None:
        unresolved.append("selected_encoder_depth")
    if family is None or size is None:
        unresolved.append("selected_decoder")
    pose_enabled = supervision in {"kp+pose", "all+pose"}
    all_targets = supervision in {"all", "all+pose"}
    argv = _base_argv(config, run_id=run_id, seed=seed)
    argv.extend(
        [
            "data/processing="
            + (
                config.composition.processing_all
                if all_targets
                else config.composition.processing_kp
            ),
            "loss="
            + (
                config.composition.loss_pose
                if pose_enabled
                else config.composition.loss_dense
            ),
            f"model/decoder=query_{family_token}_{size_token}",
            f"model.task_encoder.depth={depth_token}",
            "model.heads.dense_targets=" + ("[kp,seg,line]" if all_targets else "[kp]"),
        ]
    )
    taps: list[int] | str
    if depth is None or family is None or size is None:
        taps = SELECTED_ENCODER_PLACEHOLDER
    else:
        taps = _decoder_taps(depth, family=family, size=size)
        rendered = _render_int_list(taps)
        argv.extend(
            [
                f"model.task_encoder.tap_indices={rendered}",
                f"model.decoder.tap_indices={rendered}",
            ]
        )
        if family == "dpt":
            factors = _dpt_factors(len(taps))
            argv.extend(
                [
                    f"model.decoder.fusion_levels={len(taps)}",
                    f"model.decoder.reassemble_factors={_render_float_list(factors)}",
                ]
            )
    return _run_record(
        run_id=run_id,
        phase="supervision_third",
        phase_order=3,
        seed=seed,
        architecture={
            "encoder_depth": depth if depth is not None else depth_token,
            "hidden_dim": config.encoder_first.hidden_dim,
            "num_heads": config.encoder_first.num_heads,
            "decoder_family": family_token,
            "decoder_size": size_token,
            "encoder_taps": taps,
        },
        supervision=supervision,
        argv=argv,
        unresolved=unresolved,
    )


def _run_record(
    *,
    run_id: str,
    phase: str,
    phase_order: int,
    seed: int,
    architecture: dict[str, JsonValue],
    supervision: str,
    argv: list[str],
    unresolved: list[str],
) -> dict[str, JsonValue]:
    return {
        "run_id": run_id,
        "phase": phase,
        "phase_order": phase_order,
        "seed": seed,
        "architecture": architecture,
        "supervision": supervision,
        "queue_ready": not unresolved,
        "unresolved": unresolved,
        "command_argv": argv,
    }


def _decoder_taps(
    depth: int,
    *,
    family: DecoderFamily,
    size: DecoderSize,
) -> list[int]:
    if depth <= 0:
        raise ValueError("Selected encoder depth must be positive.")
    if family in {"linear", "progressive"}:
        return [depth - 1]
    requested = 2 if size == "tiny" else 4
    levels = min(requested, depth)
    if levels < 2:
        raise ValueError(
            "Selected encoder depth 1 cannot resolve the multi-tap DPT decoder "
            "matrix; complete encoder selection with a DPT-compatible depth."
        )
    if levels == depth:
        return list(range(depth))
    return [round(index * (depth - 1) / (levels - 1)) for index in range(levels)]


def _dpt_factors(levels: int) -> list[float]:
    if levels < 2:
        raise ValueError("DPT requires at least two fusion levels.")
    if levels == 2:
        return [2.0, 1.0]
    if levels == 3:
        return [4.0, 2.0, 1.0]
    if levels == 4:
        return [4.0, 2.0, 1.0, 0.5]
    raise ValueError("DPT ablation supports at most four fusion levels.")


def _render_int_list(values: Sequence[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _render_float_list(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.1f}" for value in values) + "]"


def _manifest_digest(manifest: Mapping[str, JsonValue]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_ablation_manifest(
    value: Mapping[str, object],
    *,
    require_resolved: bool,
) -> None:
    """Validate exact phase completeness/order and optional selection resolution."""
    expected = {
        "schema",
        "phase_order",
        "fixed_contract",
        "selection_rules",
        "selected",
        "runs",
        "manifest_sha256",
    }
    if set(value) != expected or value["schema"] != MANIFEST_SCHEMA:
        raise ValueError("Court query ablation manifest fields/schema changed.")
    if value["phase_order"] != list(PHASE_ORDER):
        raise ValueError("Court query ablation phase order changed.")
    digest = value["manifest_sha256"]
    if not isinstance(digest, str) or digest != _manifest_digest(
        cast(Mapping[str, JsonValue], value)
    ):
        raise ValueError("Court query ablation manifest digest mismatch.")
    fixed = _mapping(value["fixed_contract"], name="manifest.fixed_contract")
    if set(fixed) != {
        "seeds",
        "epochs",
        "input_hw",
        "resize",
        "preserve_fx_fy",
        "hflip",
        "affine",
        "shear",
        "perspective",
        "training_entrypoint",
        "python_executable",
        "gpu_execution_policy",
    }:
        raise ValueError("Court query ablation fixed-contract fields changed.")
    if (
        fixed["seeds"] != [42, 43, 44]
        or fixed["epochs"] != 15
        or fixed["input_hw"] != [256, 256]
        or fixed["resize"] != "isotropic_fit_letterbox"
        or fixed["preserve_fx_fy"] is not True
        or any(
            fixed[name] is not False
            for name in ("hflip", "affine", "shear", "perspective")
        )
        or fixed["python_executable"] != ".venv/bin/python"
        or fixed["training_entrypoint"] != "src.tasks.court_detection.scripts.train"
        or fixed["gpu_execution_policy"] != "enqueue_with_repository_training_queue"
    ):
        raise ValueError("Court query ablation fixed contract changed.")
    expected_rules = {
        "encoder": {
            "reference_depth": 8,
            "tolerance_ratio": 0.05,
            "metrics": [
                "kp_mean_distance_px",
                "pose_translation_l2_m",
                "pose_rotation_geodesic_deg",
                "pose_focal_relative_error",
            ],
            "rule": (
                "smallest depth whose three-seed means are all within 5% of "
                "depth-8; use depth-8 if none"
            ),
        },
        "decoder": {
            "reference": {"family": "dpt", "size": "base"},
            "tolerance_ratio": 0.05,
            "metric": "kp_mean_distance_px",
            "rule": (
                "minimum decoder MACs, then decoder parameters, among candidates "
                "within 5% KP mean-distance degradation of DPT-base"
            ),
        },
    }
    if value["selection_rules"] != expected_rules:
        raise ValueError("Court query ablation selection rules changed.")
    selected = _mapping(value["selected"], name="manifest.selected")
    if set(selected) != {"encoder_depth", "decoder_family", "decoder_size"}:
        raise ValueError("Court query ablation selected fields changed.")
    selected_depth = selected["encoder_depth"]
    selected_family = selected["decoder_family"]
    selected_size = selected["decoder_size"]
    if selected_depth is not None and selected_depth not in {1, 2, 4, 8}:
        raise ValueError("Court query selected encoder depth is invalid.")
    decoder_partly_selected = (selected_family is None) != (selected_size is None)
    if decoder_partly_selected or (
        selected_family is not None
        and (
            selected_family not in {"linear", "progressive", "dpt"}
            or selected_size not in {"tiny", "small", "base"}
        )
    ):
        raise ValueError("Court query selected decoder identity is invalid.")
    raw_runs = value["runs"]
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ValueError("Court query ablation runs must be a sequence.")
    runs = tuple(_mapping(run, name="manifest.run") for run in raw_runs)
    if len(runs) != 51:
        raise ValueError("Court query ablation manifest must contain exactly 51 runs.")
    ids: set[str] = set()
    observed_phases: list[str] = []
    phase_counts = {"encoder_first": 0, "decoder_second": 0, "supervision_third": 0}
    for run in runs:
        _validate_run(run, require_resolved=require_resolved)
        run_id = cast(str, run["run_id"])
        if run_id in ids:
            raise ValueError("Court query ablation run IDs must be unique.")
        ids.add(run_id)
        phase = cast(str, run["phase"])
        observed_phases.append(phase)
        phase_counts[phase] += 1
    if phase_counts != {
        "encoder_first": 12,
        "decoder_second": 27,
        "supervision_third": 12,
    }:
        raise ValueError("Court query ablation phase matrix is incomplete.")
    expected_ids = [
        f"encoder-depth-{depth:02d}-seed-{seed}"
        for depth in (1, 2, 4, 8)
        for seed in (42, 43, 44)
    ]
    expected_ids.extend(
        f"decoder-{family}-{size}-seed-{seed}"
        for family in ("linear", "progressive", "dpt")
        for size in ("tiny", "small", "base")
        for seed in (42, 43, 44)
    )
    expected_ids.extend(
        f"supervision-{supervision}-seed-{seed}"
        for supervision in ("kp", "kp_pose", "all", "all_pose")
        for seed in (42, 43, 44)
    )
    if [cast(str, run["run_id"]) for run in runs] != expected_ids:
        raise ValueError("Court query ablation run identity/order matrix changed.")
    if any(bool(run["queue_ready"]) for run in runs[12:39]) is (selected_depth is None):
        raise ValueError("Decoder phase readiness disagrees with encoder selection.")
    if any(bool(run["queue_ready"]) for run in runs[39:]) is (
        selected_depth is None or selected_family is None
    ):
        raise ValueError("Supervision phase readiness disagrees with prior selections.")
    phase_numbers = [PHASE_ORDER.index(phase) for phase in observed_phases]
    if phase_numbers != sorted(phase_numbers):
        raise ValueError("Court query ablation runs are not phase ordered.")


def _validate_run(run: Mapping[str, object], *, require_resolved: bool) -> None:
    expected = {
        "run_id",
        "phase",
        "phase_order",
        "seed",
        "architecture",
        "supervision",
        "queue_ready",
        "unresolved",
        "command_argv",
    }
    if set(run) != expected:
        raise ValueError("Court query ablation run fields changed.")
    phase = run["phase"]
    if phase not in PHASE_ORDER or run["phase_order"] != PHASE_ORDER.index(phase) + 1:
        raise ValueError("Court query run phase identity/order is invalid.")
    if run["seed"] not in {42, 43, 44}:
        raise ValueError("Court query run seed is outside the fixed seed set.")
    argv = run["command_argv"]
    if (
        not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes))
        or list(argv[:3])
        != [".venv/bin/python", "-m", "src.tasks.court_detection.scripts.train"]
    ):
        raise ValueError("Court query run must contain trainer argv, not a shell job.")
    unresolved = run["unresolved"]
    if not isinstance(unresolved, Sequence) or isinstance(unresolved, (str, bytes)):
        raise ValueError("Court query run unresolved fields must be a sequence.")
    ready = run["queue_ready"]
    if type(ready) is not bool or ready is not (len(unresolved) == 0):
        raise ValueError("Court query run queue-ready state is inconsistent.")
    if require_resolved and not ready:
        raise ValueError("Complete ablation results require every run to be resolved.")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return cast(Mapping[str, object], value)


__all__ = [
    "MANIFEST_SCHEMA",
    "PHASE_ORDER",
    "SELECTED_DECODER_FAMILY_PLACEHOLDER",
    "SELECTED_DECODER_SIZE_PLACEHOLDER",
    "SELECTED_ENCODER_PLACEHOLDER",
    "build_ablation_manifest",
    "validate_ablation_manifest",
]
