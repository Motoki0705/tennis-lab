"""Executable analysis boundaries preserve arguments and fail before side effects."""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

MODULES = (
    "calibrate_ball_velocity",
    "compare_ball_anchors",
    "compare_ball_transitions",
    "compare_conditions",
    "evaluate_run",
    "report_validation",
    "render_pr_clip",
)


def test_cli_roots_resolve_shared_worktree_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.slcs.scripts import _paths
    from src.utils.configuration import PathRole

    checkout = tmp_path / "worktree"
    checkout.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    for name in ("data", "ckpt", ".cache", "third_party", "outputs"):
        (shared / name).mkdir()
        (checkout / name).symlink_to(shared / name, target_is_directory=True)
    monkeypatch.setattr(_paths, "PROJECT_ROOT", checkout)
    resolver = _paths.cli_resolver(checkout / "outputs")
    assert resolver.roots.project_root == checkout
    assert resolver.roots.data_root == shared / "data"
    assert resolver.roots.checkpoint_root == shared / "ckpt"
    assert resolver.roots.artifact_root == checkout
    assert resolver.roots.output_root == shared / "outputs"
    assert resolver.roots.cache_root == shared / ".cache"
    assert resolver.roots.external_asset_root == shared / "third_party"
    assert resolver.resolve(PathRole.OUTPUT, "slcs/evaluate/example/run") == (
        shared / "outputs/slcs/evaluate/example/run"
    )


def test_task_named_input_ancestor_does_not_shadow_output_namespace(
    tmp_path: Path,
) -> None:
    from src.tasks.slcs.scripts._paths import cli_resolver
    from src.tasks.slcs.scripts.report_validation import PATH_BOUNDARY
    from src.utils.configuration import PathRole

    output_root = tmp_path / "outputs"
    resolver = cli_resolver(output_root)
    output = resolver.resolve(PathRole.OUTPUT, "slcs/visualize/comparison/run")
    PATH_BOUNDARY.validate(
        {
            "evaluations": (
                output_root / "slcs/evaluate/candidate/run",
                output_root / "slcs/train/candidate/run",
            ),
            "output_root": output_root,
            "output": output,
        },
        resolver=resolver,
        independent_artifact_inputs=True,
    )
    assert output == (output_root / "slcs/visualize/comparison/run")


def test_cli_input_aliases_resolve_without_changing_output_authority(
    tmp_path: Path,
) -> None:
    from src.tasks.slcs.scripts._paths import cli_resolver
    from src.tasks.slcs.scripts.report_validation import PATH_BOUNDARY

    source = tmp_path / "slcs"
    source.mkdir()
    alias = tmp_path / "source_alias"
    alias.symlink_to(source, target_is_directory=True)
    root = tmp_path / "outputs"
    resolver = cli_resolver(root)
    original_roots = resolver.roots
    paths = PATH_BOUNDARY.validate(
        {
            "evaluations": (alias,),
            "output_root": root,
            "output": root / "slcs/visualize/example/run",
        },
        resolver=resolver,
        independent_artifact_inputs=True,
    )
    assert paths["evaluations"] == (source,)
    assert paths["output"] == root / "slcs/visualize/example/run"
    assert resolver.roots == original_roots
    with pytest.raises(ValueError, match="duplicate paths"):
        PATH_BOUNDARY.validate(
            {**dict(paths), "evaluations": (source, alias)},
            resolver=resolver,
            independent_artifact_inputs=True,
        )


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("missing", "missing"),
        ("unknown", "unknown"),
        ("empty", "non-empty"),
        ("scalar", "sequence"),
        ("relative", "absolute"),
        ("filesystem-root", "filesystem root"),
        ("output-escape", "outside its root"),
    ],
)
def test_cli_path_adapter_keeps_strict_boundary_checks(
    tmp_path: Path, invalid: str, message: str
) -> None:
    from src.tasks.slcs.scripts._paths import cli_resolver
    from src.tasks.slcs.scripts.report_validation import PATH_BOUNDARY

    root = tmp_path / "outputs"
    arguments: dict[str, Path | tuple[Path, ...]] = {
        "evaluations": (Path("/home/slcs-cli-fixture/eval"),),
        "output_root": root,
        "output": root / "slcs/visualize/example/run",
    }
    if invalid == "missing":
        del arguments["output"]
    elif invalid == "unknown":
        arguments["extra"] = tmp_path
    elif invalid == "empty":
        arguments["evaluations"] = ()
    elif invalid == "scalar":
        arguments["evaluations"] = tmp_path
    elif invalid == "relative":
        arguments["evaluations"] = (Path("relative"),)
    elif invalid == "filesystem-root":
        arguments["evaluations"] = (Path("/"),)
    else:
        arguments["output"] = tmp_path / "outside"
    with pytest.raises(ValueError, match=message):
        PATH_BOUNDARY.validate(
            arguments, resolver=cli_resolver(root), independent_artifact_inputs=True
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", MODULES)
def test_cpu_module_help(name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"src.tasks.slcs.scripts.{name}", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


@pytest.mark.parametrize("name", ("compare_ball_anchors", "compare_ball_transitions"))
@pytest.mark.parametrize("external_inputs", [False, True], ids=["local", "cross-root"])
def test_saved_comparison_cli_routes_explicit_inputs(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_inputs: bool,
) -> None:
    module = importlib.import_module(f"src.tasks.slcs.scripts.{name}")
    calls: list[dict[str, Any]] = []
    function = (
        "save_ball_anchor_comparison"
        if name == "compare_ball_anchors"
        else "save_ball_transition_comparison"
    )
    monkeypatch.setattr(module, function, lambda **kwargs: calls.append(kwargs))
    baseline = (
        Path("/home/slcs-cli-fixture/baseline")
        if external_inputs
        else tmp_path / "baseline"
    )
    candidate = (
        Path("/mnt/slcs-cli-fixture/candidate")
        if external_inputs
        else tmp_path / "candidate"
    )
    argv = [
        name,
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(tmp_path / "report.json"),
    ]
    if name == "compare_ball_transitions":
        argv.extend(["--fast-speed-mps", "12.5"])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert calls[0]["baseline"] == baseline
    assert calls[0]["candidate"] == candidate
    assert calls[0]["output"] == tmp_path / "report.json"
    if name == "compare_ball_transitions":
        assert calls[0]["fast_speed_mps"] == 12.5
    argv[2] = "relative"
    with pytest.raises(ValueError, match="absolute"):
        module.main()
    assert len(calls) == 1


def test_calibration_cli_preserves_train_only_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.slcs.scripts import calibrate_ball_velocity as cli

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli, "calibrate_training_run", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--output-root",
            str(tmp_path),
            "--training-run",
            "slcs/train/a/b",
            "--output",
            "slcs/analyze/a/b",
            "--velocity-scale-mps",
            "4.5",
            "--gradient-ratio",
            "0.2",
            "--seed",
            "7",
            "--batch-size",
            "3",
        ],
    )
    cli.main()
    assert calls == [
        dict(
            output_root=tmp_path,
            training_run="slcs/train/a/b",
            output="slcs/analyze/a/b",
            device="cpu",
            batch_size=3,
            seed=7,
            velocity_scale_mps=4.5,
            gradient_ratio=0.2,
        )
    ]


@pytest.mark.parametrize("external_inputs", [False, True], ids=["local", "cross-root"])
def test_report_cli_keeps_all_training_labels_and_rejects_duplicate_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_inputs: bool,
) -> None:
    from src.tasks.slcs.scripts import report_validation as cli

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "generate_report", lambda **kwargs: calls.append(kwargs))
    evaluation = (
        Path("/mnt/slcs-cli-fixture/eval") if external_inputs else tmp_path / "eval"
    )
    training = (
        Path("/home/slcs-cli-fixture/train") if external_inputs else tmp_path / "train"
    )
    argv = [
        "report",
        "--evaluation",
        f"A={evaluation}",
        "--training",
        f"A={training}",
        "--output-root",
        str(tmp_path),
        "--output",
        "slcs/visualize/a/b",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls[0]["evaluations"] == {"A": evaluation}
    assert calls[0]["training"] == {"A": training}
    argv.extend(["--evaluation", f"A={tmp_path}/other"])
    with pytest.raises(ValueError, match="unique"):
        cli.main()
    assert len(calls) == 1


@pytest.mark.parametrize("root_is_symlink", [False, True])
@pytest.mark.parametrize("external_inputs", [False, True], ids=["local", "cross-root"])
def test_render_cli_passes_typed_request_and_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_is_symlink: bool,
    external_inputs: bool,
) -> None:
    from src.tasks.slcs.scripts import render_pr_clip as cli
    from src.tasks.slcs.visualization.pr_clip import RenderRequest

    calls: list[tuple[RenderRequest, tuple[str, ...]]] = []

    def render(request: RenderRequest, *, command_line: tuple[str, ...]) -> Path:
        calls.append((request, command_line))
        return tmp_path

    monkeypatch.setattr(cli, "render", render)
    output_root = tmp_path / "declared"
    real_root = tmp_path / "shared_results" if root_is_symlink else output_root
    real_root.mkdir()
    if root_is_symlink:
        output_root.symlink_to(real_root, target_is_directory=True)
    overlay = (
        Path("/home/slcs-cli-fixture/rgb.mp4")
        if external_inputs
        else tmp_path / "rgb.mp4"
    )
    scene = (
        Path("/mnt/slcs-cli-fixture/3d.mp4") if external_inputs else tmp_path / "3d.mp4"
    )
    argv = [
        "render",
        "--overlay",
        str(overlay),
        "--scene",
        str(scene),
        "--output-root",
        str(output_root),
        "--experiment",
        "fixture",
        "--run-id",
        "one",
        "--label",
        "Fixture",
        "--model",
        "baseline",
        "--clip-id",
        "clip",
        "--camera-id",
        "cam0",
        "--checkpoint-sha256",
        "a" * 64,
        "--epoch",
        "56",
        "--start",
        "0",
        "--end",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls[0][0].epoch == 56
    assert calls[0][0].fps == 10
    assert calls[0][0].overlay == overlay
    assert calls[0][0].scene == scene
    assert calls[0][1] == tuple(argv)
    outside = tmp_path / "outside"
    outside.mkdir()
    (real_root / "fixture").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        cli.main()
    assert len(calls) == 1
    assert list(outside.iterdir()) == []
