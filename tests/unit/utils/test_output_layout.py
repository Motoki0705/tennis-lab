"""Output identities remain isolated and survive resolved config round-trips."""

import ast
import importlib
import re
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

import src.utils.hydra  # noqa: F401 -- registers the public YAML resolvers
from src.utils.configuration.errors import PathContractError
from src.utils.configuration.inventory import (
    EXPECTED_RUNTIME_BOUNDARIES,
    BoundaryKind,
    RuntimeBoundary,
)
from src.utils.configuration.output_layout import dataset_output_path, task_output_path
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize(
    "component", ["../escape", "/tmp", "a/b", "", ".", " a", "x\\y"]
)
def test_output_identity_rejects_path_components(component: str) -> None:
    with pytest.raises(PathContractError):
        task_output_path("slcs", "train", component, "seed42-v1")
    with pytest.raises(PathContractError):
        dataset_output_path("slcs", component)


def test_run_identity_is_cached_per_config_and_frozen_when_saved(
    tmp_path: Path,
) -> None:
    yaml = "a: ${tennis_run_id:}\nb: ${tennis_run_id:}\n"
    first, second = OmegaConf.create(yaml), OmegaConf.create(yaml)
    assert first.a == first.b
    assert first.a != second.a
    saved = tmp_path / "config.yaml"
    OmegaConf.save(first, saved, resolve=True)
    assert OmegaConf.load(saved).a == first.a


@pytest.mark.parametrize(
    "task", ["plcs", "blcs", "court_detection", "ball_detection", "slcs"]
)
def test_training_config_uses_one_named_run_for_artifacts_and_hydra(task: str) -> None:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks" / task / "configs"),
    ):
        cfg = compose(config_name="train", return_hydra_config=True)
    parts = cfg.run.output_dir.split("/")
    assert len(parts) == 4
    assert parts[:2] == [task, "train"]
    assert cfg.paths.artifact_root == cfg.paths.output_root
    assert (
        Path(cfg.hydra.run.dir)
        == (
            PROJECT_ROOT / cfg.paths.output_root / cfg.run.output_dir / "hydra"
        ).resolve()
    )


# Read the actual decorator so a newly registered CLI cannot silently escape this
# CPU-only smoke suite or keep using an unrelated entry point's log directory.
_TASK_BOUNDARIES = tuple(
    boundary
    for boundary in EXPECTED_RUNTIME_BOUNDARIES
    if boundary.module.startswith("src.tasks.") and boundary.kind == BoundaryKind.HYDRA
)


def _compose_boundary(boundary: RuntimeBoundary, overrides: list[str]) -> DictConfig:
    source = PROJECT_ROOT / Path(*boundary.module.split(".")).with_suffix(".py")
    decorators = [
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hydra_main"
    ]
    assert len(decorators) == 1
    arguments = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in decorators[0].keywords
        if keyword.arg in {"config_path", "config_name"}
    }
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((source.parent / arguments["config_path"]).resolve()),
    ):
        return compose(
            config_name=arguments["config_name"],
            overrides=overrides,
            return_hydra_config=True,
        )


def _expected_kind(module: str) -> str:
    name = module.rsplit(".", 1)[1]
    if name == "materialize_targets" or name.startswith("precompute"):
        return "precompute"
    if name.startswith("train"):
        return "train"
    if name == "eval" or name.startswith("evaluate"):
        return "evaluate"
    if name.startswith("analyze") or name == "visualize_rotation_error_samples":
        return "analyze"
    if name.startswith(("preview", "visualize", "predict")):
        return "visualize"
    return "generate"


@pytest.mark.parametrize(
    "separate_roots", [False, True], ids=["defaults", "separate-roots"]
)
@pytest.mark.parametrize(
    "boundary", _TASK_BOUNDARIES, ids=lambda b: b.module.removeprefix("src.tasks.")
)
def test_all_task_cli_output_contracts(
    boundary: RuntimeBoundary,
    separate_roots: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = []
    if separate_roots:
        overrides = [
            f"paths.{role.value}_root={tmp_path / role.value}"
            for role in (
                PathRole.DATA,
                PathRole.OUTPUT,
                PathRole.ARTIFACT,
                PathRole.CHECKPOINT,
            )
        ]
    if "extract_gvhmr_motions" in boundary.module:
        overrides += [
            "dataset=meiji_3cam",
            f"paths.data_root={tmp_path / 'dataset-root'}",
        ]
    if "clip_and_predict_youtube_dataset" in boundary.module:
        overrides += ["workflow.video_id=smoke-video"]
    if boundary.module == "src.tasks.blcs.scripts.evaluate_real":
        overrides += ["evaluation.checkpoint=smoke/model.ckpt"]
    if boundary.module == "src.tasks.player_detection.scripts.export_checkpoint":
        overrides += [
            "export.lightning_checkpoint=smoke/model.ckpt",
            "export.destination=player_detection/smoke.pth",
        ]
    if boundary.module == "src.tasks.slcs.scripts.generate_dataset":
        overrides += ["data.dataset_root=smoke/clips"]
    cfg = _compose_boundary(boundary, overrides)
    roots = RuntimePathRoots.from_mapping(
        cast(dict[str, object], OmegaConf.to_container(cfg.paths, resolve=True)),
        repository_root=PROJECT_ROOT,
    )
    if boundary.module == "src.tasks.blcs.scripts.evaluate_real":
        # This migrated offline evaluator owns its immutable run directory and
        # saves config/metrics itself. Hydra must not pre-create it or leak logs
        # into CWD before the runtime's existing-run guard executes.
        assert cfg.hydra.run.dir == "."
        assert cfg.hydra.output_subdir is None
        assert cfg.hydra.job.chdir is False
        for logging_cfg in (cfg.hydra.job_logging, cfg.hydra.hydra_logging):
            assert "handlers" not in logging_cfg
            assert logging_cfg.disable_existing_loggers is True
        run = Path(cfg.run.output_dir)
    else:
        log = Path(cfg.hydra.run.dir).relative_to(roots.output_root)
        assert log.name == "hydra"
        run = log.parent
    assert len(run.parts) == 4
    assert run.parts[:2] == (boundary.domain, _expected_kind(boundary.module))
    assert re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z-[a-f0-9]{8}", run.parts[3])
    job = deepcopy(cfg)
    with open_dict(job):
        del job["hydra"]
    resolved: list[tuple[PathRole, tuple[str | Path, ...], Path]] = []
    original = PathResolver.resolve

    def record(resolver: PathResolver, role: PathRole, *parts: str | Path) -> Path:
        result: Path = original(resolver, role, *parts)
        resolved.append((role, parts, result))
        return result

    monkeypatch.setattr(PathResolver, "resolve", record)
    assert boundary.validator_callable is not None
    module, name = boundary.validator_callable.rsplit(".", 1)
    validator = getattr(importlib.import_module(module), name)
    if "extract_gvhmr_motions" in boundary.module:
        (roots.data_root / str(job.dataset.root)).mkdir(parents=True)
        # Only the external model/checkpoint availability check is replaced;
        # dataset, selection and output-role validation still execute.
        with patch("src.tasks.plcs.motion.extraction_config.load_model_runtime"):
            validator(job)
    else:
        validator(job)
    if boundary.module != "src.tasks.blcs.generate_dataset.api_server.__main__":
        assert resolved, "Boundary must exercise typed path resolution"
    for role, parts, result in resolved:
        assert result.is_relative_to(roots.root(role)), (role, parts, result)
    for key in (
        "run.output_dir",
        "preview.output_dir",
        "generate_line_masks.preview_dir",
        "evaluate.output_dir",
        "predict.output_dir",
        "analysis.output_dir",
        "analyze.output_dir",
        "visualization.save",
        "convert.output_dir",
    ):
        fragment = OmegaConf.select(job, key)
        if fragment is None:
            continue
        expected_role = PathRole.OUTPUT
        if key == "visualization.save" and boundary.domain in {
            "ball_detection",
            "court_detection",
        }:
            expected_role = PathRole.ARTIFACT
        if key == "convert.output_dir" or (
            key == "run.output_dir"
            and boundary.module.rsplit(".", 1)[1]
            in {"generate_dataset", "extract_gvhmr_motions"}
        ):
            expected_role = PathRole.DATA
        assert any(
            role == expected_role and parts[0] == fragment
            for role, parts, _ in resolved
        ), (key, fragment, expected_role)
        if expected_role != PathRole.DATA:
            assert Path(str(fragment)).parts[:4] == run.parts


@pytest.mark.parametrize("task", ["blcs", "plcs"])
@pytest.mark.parametrize("mode", ["visualize", "predict"])
def test_visualize_mode_shares_log_identity_with_artifact(
    task: str, mode: str, tmp_path: Path
) -> None:
    boundary = next(
        b for b in _TASK_BOUNDARIES if b.module == f"src.tasks.{task}.scripts.visualize"
    )
    cfg = _compose_boundary(
        boundary,
        [
            f"visualization.mode={mode}",
            "visualization.checkpoint=smoke/model.ckpt",
            f"paths.output_root={tmp_path / 'logs'}",
            f"paths.artifact_root={tmp_path / 'media'}",
        ],
    )
    log = Path(cfg.hydra.run.dir).relative_to(tmp_path / "logs")
    assert Path(cfg.visualization.save).parent == log.parent
    assert log.parts[:3] == (task, "visualize", mode)
    job = deepcopy(cfg)
    with open_dict(job):
        del job["hydra"]
    runtime_module = importlib.import_module(
        f"src.tasks.{task}.visualization.orchestrator"
    )
    runtime = runtime_module.build_runtime_config(job)
    assert runtime.save == tmp_path / "logs" / cfg.visualization.save


@pytest.mark.parametrize(
    "name,kind,output_keys",
    [
        ("pipeline", "generate", ("output_directory",)),
        ("visualization", "visualize", ("output", "preview_output")),
        ("visualize_tasks", "visualize", ("output_directory",)),
        ("clip_studio", "generate", ()),
        ("generate_dataset", "generate", ()),
    ],
)
def test_tennis_scene_logs_and_outputs_have_named_run(
    name: str, kind: str, output_keys: tuple[str, ...], tmp_path: Path
) -> None:
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")
    ):
        cfg = compose(
            config_name=name,
            overrides=[f"paths.output_root={tmp_path / 'runs'}"],
            return_hydra_config=True,
        )
    log = Path(cfg.hydra.run.dir).relative_to(tmp_path / "runs")
    assert len(log.parts) == 5
    assert log.parts[:3] == ("tennis_scene", kind, name)
    assert log.name == "hydra"
    for key in output_keys:
        assert Path(str(OmegaConf.select(cfg, key))).parts[:4] == log.parts[:4]


def test_evaluation_manifest_resolves_output_and_checkpoint_roots(
    tmp_path: Path,
) -> None:
    from src.tasks.ball_detection.evaluation.contracts import load_evaluation_manifest

    boundary = next(
        b for b in _TASK_BOUNDARIES if b.module.endswith(".evaluate_manifest")
    )
    cfg = _compose_boundary(
        boundary,
        [
            f"paths.output_root={tmp_path / 'runs'}",
            f"paths.checkpoint_root={tmp_path / 'weights'}",
            f"paths.artifact_root={tmp_path / 'media'}",
        ],
    )
    roots = RuntimePathRoots.from_mapping(
        cast(dict[str, object], OmegaConf.to_container(cfg.paths, resolve=True)),
        repository_root=PROJECT_ROOT,
    )
    manifest = load_evaluation_manifest(
        PROJECT_ROOT / cfg.manifest_path, resolver=PathResolver(roots)
    )
    relative = manifest.output_dir.relative_to(tmp_path / "runs")
    assert len(relative.parts) == 4
    assert relative.parts[:3] == ("ball_detection", "evaluate", "detector_comparison")
    assert all(
        model.checkpoint.is_relative_to(tmp_path / "weights")
        for model in manifest.models
    )


def test_mixed_training_snapshot_preserves_roots_identity_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner
    from src.tasks.court_detection.training.runner_mixed import (
        MixedCourtDetectionTrainingRunner,
        resolve_mixed_training_config,
    )

    boundary = next(b for b in _TASK_BOUNDARIES if b.module.endswith(".train_mixed"))
    cfg = _compose_boundary(boundary, [f"paths.output_root={tmp_path / 'runs'}"])
    log = Path(cfg.hydra.run.dir)
    with open_dict(cfg):
        del cfg["hydra"]
    output_dir = log.parent
    output_dir.mkdir(parents=True)

    def save_without_training(
        runner: MixedCourtDetectionTrainingRunner, standard: DictConfig
    ) -> None:
        runner.save_config(standard, output_dir)

    monkeypatch.setattr(CourtDetectionTrainingRunner, "run", save_without_training)
    MixedCourtDetectionTrainingRunner().run(cfg)
    saved = cast(DictConfig, OmegaConf.load(output_dir / "config.yaml"))
    assert saved.run.output_dir == cfg.run.output_dir
    assert saved.mixed == cfg.mixed
    assert Path(saved.paths.output_root) == tmp_path / "runs"
    assert all(Path(str(root)).is_absolute() for root in saved.paths.values())
    assert "${" not in (output_dir / "config.yaml").read_text()
    replay, mixed = resolve_mixed_training_config(saved)
    assert replay.run.output_dir == cfg.run.output_dir
    assert mixed is not None
