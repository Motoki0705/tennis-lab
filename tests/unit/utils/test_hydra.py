from __future__ import annotations

from src.utils.hydra import hydra_main


def test_hydra_main_returns_a_decorator() -> None:
    decorator = hydra_main(version_base="1.3", config_path=None)
    assert callable(decorator)

    def sample(cfg: object) -> object:
        return cfg

    wrapped = decorator(sample)
    assert callable(wrapped)


def test_scripts_use_the_shared_wrapper() -> None:
    """Hydra handlers retain the shared wrapper behind the public CLI router."""
    from src.tasks.ball_detection.evaluation import heatmap_cli as ball_eval
    from src.tasks.plcs.scripts import train as plcs_train

    assert ball_eval.hydra_main is hydra_main
    assert plcs_train.hydra_main is hydra_main


def test_relative_config_path_survives_dispatch(tmp_path, monkeypatch, capsys) -> None:
    """An entry point imported by dispatch_cli finds configs that are not a Python package."""
    from src.utils.configuration.cli import dispatch_cli

    package = tmp_path / "dispatchpkg"
    (package / "entry").mkdir(parents=True)
    (package / "configs").mkdir()
    (package / "__init__.py").write_text("")
    (package / "entry" / "__init__.py").write_text("")
    (package / "configs" / "job.yaml").write_text("marker: from-configs\n")
    (package / "entry" / "runner.py").write_text(
        "from src.utils.hydra import hydra_main\n\n\n"
        "@hydra_main(config_path='../configs', config_name='job', version_base='1.3')\n"
        "def main(cfg):\n"
        "    print('marker=' + cfg.marker)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "cli", "--stage", "run", "hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=none",
        "hydra/hydra_logging=none",
    ])

    dispatch_cli("stage", {"run": "dispatchpkg.entry.runner"}, description="test")

    assert "marker=from-configs" in capsys.readouterr().out


def test_hybrid_training_stage_composes_through_the_train_router(monkeypatch, capsys) -> None:
    from src.tasks.ball_detection.scripts import train

    monkeypatch.setattr("sys.argv", ["train", "--stage", "hybrid", "--cfg", "job"])

    train.main()

    assert "windows_per_epoch" in capsys.readouterr().out
