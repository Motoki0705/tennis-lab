import textwrap

import pytest

from src.wasb.scripts.generate_game import build_args_from_config, load_config


def test_build_args_batch_mode_minimal() -> None:
    config = {
        "mode": "batch",
        "video_dir": "data/tennis/raw",
    }

    args = build_args_from_config(config)

    assert args.video_dir == "data/tennis/raw"
    assert args.output_dir == "data/tennis"
    assert args.status is False
    assert args.reset_failed is False
    assert args.reset_all is False
    assert args.generate_samples == []
    assert args.apply_clip_selection == []
    assert args.apply_completion == []
    assert args.model == "wasb"
    assert args.checkpoint.endswith("pretrained/wasb_tennis_best.pth.tar")


def test_build_args_single_video_requires_fields() -> None:
    config = {
        "mode": "single_video",
        "video": "video.mp4",
        # "output" is intentionally missing
    }

    with pytest.raises(ValueError):
        build_args_from_config(config)


def test_build_args_invalid_mode() -> None:
    config = {
        "mode": "invalid_mode",
    }

    with pytest.raises(ValueError):
        build_args_from_config(config)


def test_load_config_with_custom_path(tmp_path) -> None:
    yaml_path = tmp_path / "generate_game.yaml"
    yaml_content = textwrap.dedent(
        """
        mode: status
        output_dir: data/tennis
        """
    )
    yaml_path.write_text(yaml_content)

    config = load_config(yaml_path)

    assert config["mode"] == "status"
    assert config["output_dir"] == "data/tennis"
