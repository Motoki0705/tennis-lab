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
from omegaconf.errors import InterpolationResolutionError

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


# Expectations name the semantic validator contract, not a module basename.
# A new contract must declare its expected output kind rather than defaulting to generate.
_OUTPUT_KIND_CONTRACTS = {
    "train": (
        "ball_refiner_3d.train", "ball.train", "ball.hybrid", "blcs.train", "court_detection.train",
        "player_detection.train", "plcs.train", "slcs.train",
    ),
    "generate": (
        "ball_refiner_3d.generate_dataset", "ball_detection.generate_dataset",
        "blcs.api_server", "blcs.generate_dataset", "blcs.generate_dataset_samples",
        "player_detection.generate_dataset", "player_detection.export_checkpoint",
        "plcs.generate_dataset", "plcs.generate_dataset_samples",
        "slcs.generate_dataset", "slcs.make_splits",
    ),
    "evaluate": ("ball.eval", "ball.evaluate_manifest", "player_detection.evaluate", "slcs.evaluate"),
    "visualize": (
        "ball.preview", "ball.visualize", "ball.learning_stages", "blcs.preview_augmentation", "blcs.visualize",
        "ball.annotation_correction",
        "court_detection.preview_augmentation", "court_detection.visualize",
        "court_detection.learning_stages",
        "player_detection.preview_dataset", "plcs.preview_augmentation", "plcs.visualize", "slcs.predict_clip",
    ),
    "analyze": (
        "plcs.analyze_angle_velocity", "plcs.analyze_dataset_distribution", "plcs.analyze_loss_dominance",
        "plcs.analyze_rotation_error_samples", "slcs.analyze_predictions",
    ),
    "precompute": ("slcs.precompute_dino_tokens",),
}
_EXPECTED_KIND_BY_CONTRACT = {
    contract: kind for kind, contracts in _OUTPUT_KIND_CONTRACTS.items() for contract in contracts
}


def _expected_kind(boundary: RuntimeBoundary) -> str:
    assert boundary.validator_key is not None
    return _EXPECTED_KIND_BY_CONTRACT[boundary.validator_key]


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
    if boundary.module == "src.tasks.blcs.scripts.evaluate_real":
        overrides += ["evaluation.checkpoint=smoke/model.ckpt"]
    if boundary.module == "src.tasks.player_detection.scripts.export_checkpoint":
        overrides += [
            "export.lightning_checkpoint=smoke/model.ckpt",
            "export.destination=player_detection/smoke.pth",
        ]
    if boundary.module == "src.tasks.slcs.scripts.generate_dataset":
        overrides += ["data.dataset_root=smoke/clips"]
    if boundary.validator_key == "ball.learning_stages":
        overrides += ["input_panel=schematic"]
    if boundary.module == "src.tasks.ball_detection.training.hybrid.runner":
        overrides += [
            "manifest=smoke/windows.json", "initialization.mode=query",
            "initialization.checkpoint=smoke/query.pt", "presence_threshold=0.5",
            "loss.heatmap_weight=1.0", "loss.presence_weight=1.0",
            "freeze.encoder_epochs=0", "freeze.query_epochs=0",
            "absence.negative_kinds=[reviewed_empty,out_of_frame]",
            "absence.window_weights.play=0.5", "absence.window_weights.absent=0.5",
            "absence.window_weights.nonplay=0.0", "epochs=1", "windows_per_epoch=4",
            "learning_rate=0.001", "warmup_updates=0", "seed=42", "device=cpu",
            "runtime.precision=fp32", "runtime.compile_mode=off",
            "runtime.jpeg_decoder=opencv", "runtime.input_verification=lazy",
        ]
    if boundary.validator_key == "ball.annotation_correction":
        overrides += [
            f"paths.annotation_root={tmp_path / 'annotation-input'}",
            f"paths.campaign_dir={tmp_path / 'campaign-input'}",
        ]
    cfg = _compose_boundary(boundary, overrides)
    role_paths = cast(dict[str, object], OmegaConf.to_container(cfg.paths, resolve=True))
    if boundary.validator_key == "ball.annotation_correction":
        # Legacy annotation inputs are explicitly absolute paths, not members
        # of the seven shared role roots. The CLI validates these two itself.
        del role_paths["annotation_root"], role_paths["campaign_dir"]
    roots = RuntimePathRoots.from_mapping(
        role_paths,
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
    assert run.parts[:2] == (boundary.domain, _expected_kind(boundary))
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
        ("blender_export", "visualize", ("output",)),
        ("export_blender_diffusion", "visualize", ("output",)),
        ("export_blender_rays", "visualize", ("output",)),
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


def test_evaluation_manifest_keeps_artifact_models_and_pretrained_roots_separate(
    tmp_path: Path,
) -> None:
    from src.tasks.ball_detection.evaluation.contracts import load_evaluation_manifest

    boundary = next(
        b for b in _TASK_BOUNDARIES if b.validator_key == "ball.evaluate_manifest"
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
        model.checkpoint.is_relative_to(tmp_path / "media")
        for model in manifest.models
    )
    assert roots.checkpoint_root == tmp_path / "weights"


def test_mixed_training_snapshot_preserves_roots_identity_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.court_detection.training.runner import (
        CourtDetectionTrainingRunner,
        resolve_training_config,
    )

    boundary = next(
        b
        for b in _TASK_BOUNDARIES
        if b.module == "src.tasks.court_detection.scripts.train"
    )
    cfg = _compose_boundary(boundary, [f"paths.output_root={tmp_path / 'runs'}"])
    log = Path(cfg.hydra.run.dir)
    with open_dict(cfg):
        del cfg["hydra"]
    output_dir = log.parent
    output_dir.mkdir(parents=True)

    def save_without_training(
        runner: CourtDetectionTrainingRunner, standard: DictConfig
    ) -> None:
        runner.save_config(standard, output_dir)

    monkeypatch.setattr(CourtDetectionTrainingRunner, "run", save_without_training)
    CourtDetectionTrainingRunner().run(cfg)
    saved = cast(DictConfig, OmegaConf.load(output_dir / "config.yaml"))
    assert saved.run.output_dir == cfg.run.output_dir
    assert saved.mixed == cfg.mixed
    assert Path(saved.paths.output_root) == tmp_path / "runs"
    assert all(Path(str(root)).is_absolute() for root in saved.paths.values())
    assert "${" not in (output_dir / "config.yaml").read_text()
    replay, mixed = resolve_training_config(saved)
    assert replay.run.output_dir == cfg.run.output_dir
    assert mixed is not None


def test_output_root_named_like_the_task_is_rejected_with_the_colliding_root(
    tmp_path: Path,
) -> None:
    # Regression for queue job i931-v2-dataset-meiji-one-clip-20260927:
    # paths.output_root=<report>/slcs made the precompute log directory
    # <root>/slcs/precompute/... fail on the reserved basename 'slcs' with no
    # hint that the override itself was the cause.
    config_dir = str(PROJECT_ROOT / "src/tasks/slcs/configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="precompute_dino_tokens",
            overrides=[f"paths.output_root={tmp_path / 'slcs'}"],
            return_hydra_config=True,
        )
        with pytest.raises(
            InterpolationResolutionError, match="configured output_root"
        ) as error:
            _ = cfg.hydra.run.dir
    assert str(tmp_path / "slcs") in str(error.value)

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="precompute_dino_tokens",
            overrides=[f"paths.output_root={tmp_path / 'report'}"],
            return_hydra_config=True,
        )
    log = Path(cfg.hydra.run.dir).relative_to(tmp_path / "report")
    assert log.parts[:3] == ("slcs", "precompute", "precompute_dino_tokens")
