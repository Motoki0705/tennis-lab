"""Negative matrix for synthetic runtime configuration boundaries."""

from __future__ import annotations

import ast
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.synthetic_data_generation.configuration import validate_config
from src.utils.configuration import (
    BoundaryPathField,
    ConfigurationError,
    NonHydraPathBoundary,
    PathContractError,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"
_BOUNDARIES = {
    "synthetic.dataset.pipeline": "dataset/pipeline",
    "synthetic.dataset.blcs.feature_fit": "dataset/blcs_feature_fit",
    "synthetic.alignment.infer_ground_line_map": "alignment/infer_ground_line_map",
    "synthetic.alignment.fit_ground_courts": "alignment/fit_ground_courts",
    "synthetic.alignment.calibrate_court_alignment": (
        "alignment/calibrate_court_alignment"
    ),
    "synthetic.alignment.export_scene_provider": "alignment/export_scene_provider",
    "synthetic.alignment.geometry_bridge": "alignment/geometry_bridge",
}
_RAW_BOUNDARY_MODULES = (
    "src.synthetic_data_generation.alignment.scene_provider.geometry_bridge",
    "src.synthetic_data_generation.dataset.blcs.components.asset_preparation",
    "src.synthetic_data_generation.dataset.blcs.components.calibration_import",
    "src.synthetic_data_generation.dataset.blcs.components.feature_fit_fixture",
    "src.synthetic_data_generation.dataset.blcs.components.procedural_ball_asset_builder",
    "src.synthetic_data_generation.dataset.blcs.rendering.feature_fit",
    "src.synthetic_data_generation.dataset.blcs.rendering.nht",
    "src.synthetic_data_generation.dataset.court.components.camera_sampling.orbit_plan",
    "src.synthetic_data_generation.dataset.court.components.camera_sampling.support_probe",
    "src.synthetic_data_generation.dataset.court.rendering.nht",
    "src.synthetic_data_generation.dataset.court.rendering.orbit_preview",
    "src.synthetic_data_generation.dataset.court.visualization.camera_support",
    "src.synthetic_data_generation.dataset.court.visualization.dataset_preview",
    "src.synthetic_data_generation.dataset.plcs.components.avatar_asset_builder",
    "src.synthetic_data_generation.dataset.plcs.components.scene_plan_builder",
    "src.synthetic_data_generation.dataset.plcs.rendering.avatar_fit",
    "src.synthetic_data_generation.dataset.plcs.rendering.nht",
    "src.synthetic_data_generation.dataset.plcs.visualization.avatar_control_comparison",
    "src.synthetic_data_generation.dataset.plcs.visualization.dataset_preview",
    "src.synthetic_data_generation.rendering.nht.composition_smoke",
    "src.synthetic_data_generation.rendering.nht.runtime_probe",
)


def _compose(name: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        return compose(config_name=name)


def _must_reject(name: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except (ConfigurationError, PathContractError, TypeError, ValueError):
        return
    raise AssertionError(f"Synthetic negative validation case was accepted: {name}")


def _source_boundary(module_name: str) -> NonHydraPathBoundary:
    path = PROJECT_ROOT / (module_name.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declaration = next(
        (
            statement.value
            for statement in tree.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "PATH_BOUNDARY"
                for target in (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else (statement.target,)
                )
            )
            and isinstance(statement.value, ast.Call)
        ),
        None,
    )
    if declaration is None:
        raise AssertionError(f"Missing PATH_BOUNDARY declaration: {module_name}")
    keywords = {item.arg: item.value for item in declaration.keywords if item.arg}
    name_node = keywords["name"]
    fields_node = keywords["fields"]
    if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
        raise AssertionError(f"Boundary name is not literal: {module_name}")
    if not isinstance(fields_node, ast.Tuple):
        raise AssertionError(f"Boundary fields are not a literal tuple: {module_name}")
    fields: list[BoundaryPathField] = []
    for node in fields_node.elts:
        if not isinstance(node, ast.Call) or len(node.args) < 4:
            raise AssertionError(f"Boundary field is not explicit: {module_name}")
        field_name, role_node, direction_node, kind_node = node.args[:4]
        if (
            not isinstance(field_name, ast.Constant)
            or not isinstance(field_name.value, str)
            or not isinstance(role_node, ast.Attribute)
            or not isinstance(direction_node, ast.Attribute)
            or not isinstance(kind_node, ast.Attribute)
        ):
            raise AssertionError(
                f"Boundary field metadata is not literal: {module_name}"
            )
        flags = {
            item.arg: item.value.value
            for item in node.keywords
            if item.arg and isinstance(item.value, ast.Constant)
        }
        fields.append(
            BoundaryPathField(
                field_name.value,
                PathRole[role_node.attr],
                PathDirection[direction_node.attr],
                PathKind[kind_node.attr],
                must_exist=bool(flags.get("must_exist", False)),
                many=bool(flags.get("many", False)),
            )
        )
    return NonHydraPathBoundary(name_node.value, tuple(fields))


def _raw_boundary_matrix() -> tuple[str, ...]:
    passed: list[str] = []
    with TemporaryDirectory(prefix="synthetic-boundaries-") as raw_root:
        root = Path(raw_root).resolve()
        roots = RuntimePathRoots(
            project_root=root / "project",
            data_root=root / "data",
            checkpoint_root=root / "checkpoint",
            artifact_root=root / "artifact",
            output_root=root / "output",
            cache_root=root / "cache",
            external_asset_root=root / "external",
        )
        for role in PathRole:
            roots.root(role).mkdir(parents=True, exist_ok=True)
        resolver = PathResolver(roots)
        for module_name in _RAW_BOUNDARY_MODULES:
            boundary = _source_boundary(module_name)
            arguments: dict[str, object] = {}
            token = boundary.name.replace(".", "-")
            for field in boundary.fields:
                values: list[Path] = []
                count = 2 if field.many else 1
                for index in range(count):
                    candidate = roots.root(field.role) / token / f"{field.name}-{index}"
                    if field.must_exist and field.kind is PathKind.DIRECTORY:
                        candidate.mkdir(parents=True)
                    elif field.must_exist:
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        candidate.write_bytes(b"fixture")
                    values.append(candidate)
                arguments[field.name] = values if field.many else values[0]
            boundary.validate(arguments, resolver=resolver)
            passed.append(f"{boundary.name}:canonical")

            first = boundary.fields[0]
            missing = dict(arguments)
            del missing[first.name]
            _must_reject(
                f"{boundary.name}:missing",
                partial(boundary.validate, missing, resolver=resolver),
            )
            unknown = {**arguments, "__unknown__": roots.project_root / "unknown"}
            _must_reject(
                f"{boundary.name}:unknown",
                partial(boundary.validate, unknown, resolver=resolver),
            )
            for suffix, invalid in (
                ("wrong-type", 3),
                ("relative", Path("relative")),
                ("escape", roots.root(first.role).parent / "outside"),
            ):
                mutated = dict(arguments)
                mutated[first.name] = [invalid] if first.many else invalid
                _must_reject(
                    f"{boundary.name}:{suffix}",
                    partial(boundary.validate, mutated, resolver=resolver),
                )
                passed.append(f"{boundary.name}:{suffix}")
            passed.extend((f"{boundary.name}:missing", f"{boundary.name}:unknown"))
    return tuple(passed)


def run_validation_matrix() -> tuple[str, ...]:
    """Run canonical and invalid cases without model, filesystem, or CUDA work."""
    passed: list[str] = []
    for boundary, config_name in _BOUNDARIES.items():
        canonical = _compose(config_name)
        validate_config(boundary, canonical)
        passed.append(f"{boundary}:canonical")

        unknown = deepcopy(canonical)
        with open_dict(unknown):
            unknown["typo_root"] = "data"
        _must_reject(
            f"{boundary}:unknown",
            partial(validate_config, boundary, unknown),
        )
        passed.append(f"{boundary}:unknown")

        raw_missing = OmegaConf.to_container(canonical, resolve=False)
        if not isinstance(raw_missing, dict):
            raise AssertionError(
                "Canonical synthetic config did not compose to a mapping."
            )
        missing = OmegaConf.create(raw_missing)
        if not isinstance(missing, DictConfig):
            raise AssertionError("Synthetic missing-key fixture is not a DictConfig.")
        del missing["roots"]
        _must_reject(
            f"{boundary}:missing",
            partial(validate_config, boundary, missing),
        )
        passed.append(f"{boundary}:missing")

        wrong_type = deepcopy(canonical)
        wrong_type.roots.data_root = 3
        _must_reject(
            f"{boundary}:wrong-type",
            partial(validate_config, boundary, wrong_type),
        )
        passed.append(f"{boundary}:wrong-type")

    conflicting_renderer = _compose("dataset/pipeline")
    conflicting_renderer.renderer.mode = "execute"
    _must_reject(
        "synthetic.dataset.pipeline:renderer-conflict",
        lambda: validate_config("synthetic.dataset.pipeline", conflicting_renderer),
    )
    passed.append("synthetic.dataset.pipeline:renderer-conflict")

    escaping_path = _compose("dataset/pipeline")
    escaping_path.paths.dataset_root = "../outside"
    _must_reject(
        "synthetic.dataset.pipeline:path-escape",
        lambda: validate_config("synthetic.dataset.pipeline", escaping_path),
    )
    passed.append("synthetic.dataset.pipeline:path-escape")

    semantic_cases = (
        (
            "synthetic.dataset.pipeline",
            "dataset/pipeline",
            "renderer.command",
            ["renderer", "{ouptut}"],
            "unknown-placeholder",
        ),
        (
            "synthetic.dataset.blcs.feature_fit",
            "dataset/blcs_feature_fit",
            "feature_lr",
            0.0,
            "nonpositive-feature-lr",
        ),
        (
            "synthetic.alignment.infer_ground_line_map",
            "alignment/infer_ground_line_map",
            "ground_plane.min_camera_height",
            1.0,
            "reversed-camera-height",
        ),
        (
            "synthetic.alignment.infer_ground_line_map",
            "alignment/infer_ground_line_map",
            "line_projection.probability_threshold",
            1.5,
            "invalid-probability",
        ),
        (
            "synthetic.alignment.fit_ground_courts",
            "alignment/fit_ground_courts",
            "fit.proposal_fraction",
            0.5,
            "invalid-proposal-mixture",
        ),
        (
            "synthetic.alignment.calibrate_court_alignment",
            "alignment/calibrate_court_alignment",
            "checkpoint_best_val_dice",
            0.1,
            "reversed-checkpoint-quality",
        ),
        (
            "synthetic.alignment.calibrate_court_alignment",
            "alignment/calibrate_court_alignment",
            "local_refit.scale_relative_tolerance",
            1.0,
            "invalid-local-refit-tolerance",
        ),
        (
            "synthetic.alignment.export_scene_provider",
            "alignment/export_scene_provider",
            "factor",
            0,
            "nonpositive-export-factor",
        ),
        (
            "synthetic.alignment.geometry_bridge",
            "alignment/geometry_bridge",
            "request",
            " ",
            "blank-cache-path",
        ),
    )
    for boundary, config_name, key, invalid, label in semantic_cases:
        invalid_config = _compose(config_name)
        OmegaConf.update(invalid_config, key, invalid, merge=False)
        _must_reject(
            f"{boundary}:{label}",
            partial(validate_config, boundary, invalid_config),
        )
        passed.append(f"{boundary}:{label}")
    passed.extend(_raw_boundary_matrix())
    return tuple(passed)


__all__ = ["run_validation_matrix"]
