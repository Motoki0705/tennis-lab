import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import yaml

from src.tasks.ball_detection.scripts import train_best_mdd_posttraining
from src.tasks.ball_detection.scripts.train_cnn_candidate import (
    run_serial_candidate,
    training_command,
    validate_serial_prefix,
)
from src.tasks.ball_detection.training.heatmap_pretraining.diagnostics import (
    batch_identity,
    record_failure,
)
from src.utils.checksum import dual_sha256
from tests.unit.tasks.ball_detection.models.test_mdd_pretrain import small_config


def test_failure_receipt_uses_only_cpu_metadata_and_keeps_original_exception(tmp_path: Path) -> None:
    metadata = batch_identity(dict(clip_id=["clip-a"], start=[10], frame_step=[4], frame_indices=torch.tensor([[10, 14, 18]])))
    context = dict(epoch=0, completed_updates=5800, attempted_update=5801, surface_phase="gradient_norm", batch=metadata)
    error = RuntimeError("CUDA unknown error")
    with pytest.raises(RuntimeError) as caught, patch("torch.cuda.synchronize", side_effect=AssertionError("broken CUDA context")), \
            record_failure(tmp_path, context):
        raise error
    assert caught.value is error
    receipt = json.loads((tmp_path / "failure.json").read_text())
    assert receipt["batch"]["frame_indices"] == [[10, 14, 18]]
    assert receipt["completed_updates"] == 5800
    assert "not proof" in receipt["caveat"]
    assert "CUDA unknown error" in receipt["traceback"]


def test_metadata_rejects_non_cpu_tensor_without_transferring_it() -> None:
    with pytest.raises(ValueError, match="CPU"):
        batch_identity(dict(clip_id=[], start=[], frame_step=[], frame_indices=torch.empty(1, device="meta")))


def test_receipt_io_failure_does_not_hide_the_training_error(tmp_path: Path) -> None:
    error = RuntimeError("training failure")
    with patch.object(Path, "write_text", side_effect=OSError("disk full")), pytest.raises(RuntimeError) as caught, \
            record_failure(tmp_path, {}):
        raise error
    assert caught.value is error
    assert "disk full" in error.__notes__[0]


def serial_arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(output=tmp_path / "run", smoke_output=tmp_path / "probe", manifest=tmp_path / "manifest",
                              model_config=tmp_path / "model", prefetch_mode="serial")


def test_serial_prefix_failure_cannot_launch_full_training(tmp_path: Path) -> None:
    args = serial_arguments(tmp_path)
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "diagnostic")) as execute, \
            pytest.raises(subprocess.CalledProcessError):
        run_serial_candidate(args)
    assert execute.call_count == 1
    command = execute.call_args.args[0]
    assert "--image-prefetch" not in command
    assert command[command.index("--epochs") + 1] == "10"
    assert command[command.index("--windows-per-epoch") + 1] == "6000"
    assert command[-2:] == ["--stop-after-epoch", "0"]


def test_successful_serial_prefix_continues_its_checkpoint_without_restarting(tmp_path: Path) -> None:
    args = serial_arguments(tmp_path)

    def prefix_succeeds(command: list[str], *, check: bool) -> None:
        if "--stop-after-epoch" in command:
            args.output.mkdir()
            (args.output / "epoch-000.pt").touch()

    with patch("subprocess.run", side_effect=prefix_succeeds) as execute, \
            patch("src.tasks.ball_detection.scripts.train_cnn_candidate.validate_serial_prefix", return_value={}) as verify:
        run_serial_candidate(args)
    assert verify.call_count == 1 and execute.call_count == 2
    first, second = [call.args[0] for call in execute.call_args_list]
    assert first[:-2] == second[:-2] == training_command(args)
    assert second[-2:] == ["--resume", str(args.output / "epoch-000.pt")]
    assert (args.smoke_output / "SERIAL_PREFIX_PASSED.json").is_file()


def test_unverified_existing_prefix_cannot_trigger_a_restart(tmp_path: Path) -> None:
    args = serial_arguments(tmp_path)
    args.output.mkdir()
    with patch("subprocess.run", side_effect=AssertionError("must not launch")), pytest.raises(FileNotFoundError):
        run_serial_candidate(args)


def test_serial_prefix_checks_saved_weights_source_and_budget(tmp_path: Path) -> None:
    args = serial_arguments(tmp_path)
    args.output.mkdir()
    args.manifest.write_text("frozen manifest")
    config = asdict(small_config())
    args.model_config.write_text(yaml.safe_dump(dict(name="mdd_dpt_pretrain", **config)))
    source = {"source_sha256": {"runner.py": "known-source"}}
    saved: dict[str, Any] = dict(recipe=dict(model=config, runtime=dict(image_prefetch=False, precision="bf16", jpeg_decoder="nvjpeg"),
                             manifest_sha256=dual_sha256(args.manifest), epochs=10, windows_per_epoch=6000),
                 code=source, epoch=0, global_step=6000, state_dict={"weight": torch.ones(2)})
    checkpoint = args.output / "epoch-000.pt"
    torch.save(saved, checkpoint)
    stop = dict(stage="heatmap_pretraining_diagnostic_stop", epoch=0, global_step=6000, total_updates=60000,
                checkpoint=checkpoint.name, sha256=dual_sha256(checkpoint), complete=False)
    (args.output / "DIAGNOSTIC_STOP.json").write_text(json.dumps(stop))
    (args.output / "train.jsonl").write_text(json.dumps(dict(epoch=0, global_step=6000, train_loss=.01, grad_norm=.1,
                                                             temporal_gradient_norms=[.01, .02])) + "\n")
    with patch("src.tasks.ball_detection.scripts.train_cnn_candidate.source_identity", return_value=source):
        assert validate_serial_prefix(args)["global_step"] == 6000
        stop["total_updates"] = 6000
        (args.output / "DIAGNOSTIC_STOP.json").write_text(json.dumps(stop))
        with pytest.raises(ValueError, match="full schedule"):
            validate_serial_prefix(args)
        stop["total_updates"] = 60000
        saved["state_dict"]["weight"][0] = float("nan")
        torch.save(saved, checkpoint)
        stop["sha256"] = dual_sha256(checkpoint)
        (args.output / "DIAGNOSTIC_STOP.json").write_text(json.dumps(stop))
        with pytest.raises(ValueError, match="nonfinite"):
            validate_serial_prefix(args)


@pytest.mark.parametrize("prefetch", [False, True])
def test_posttraining_probe_uses_and_requires_the_training_prefetch_mode(tmp_path: Path, prefetch: bool) -> None:
    args = argparse.Namespace(manifest=tmp_path / "manifest", pretraining_run=tmp_path / "pretrained",
                              augmentation_config=tmp_path / "augment.yaml", output=tmp_path / "post",
                              resume=None, image_prefetch=prefetch)
    args.manifest.touch()
    args.augmentation_config.touch()
    args.pretraining_run.mkdir()
    checkpoint = args.pretraining_run / "epoch-009.pt"
    checkpoint.touch()
    probe = tmp_path / "post-probe"
    receipt = dict(checkpoint_sha256=dual_sha256(checkpoint), manifest_sha256=dual_sha256(args.manifest),
                   augmentation_sha256=dual_sha256(args.augmentation_config), image_prefetch=prefetch)

    def process(command: list[str], *, check: bool) -> None:
        if "src.tasks.ball_detection.scripts.probe_mdd_posttraining" in command:
            assert ("--image-prefetch" in command) == prefetch
            probe.mkdir()
            (probe / "PROBE_COMPLETED.json").write_text(json.dumps(receipt))

    with patch.object(train_best_mdd_posttraining, "arguments", return_value=args), \
            patch.object(train_best_mdd_posttraining, "completed_pretraining", return_value=(checkpoint, {})), \
            patch("subprocess.run", side_effect=process) as execute:
        train_best_mdd_posttraining.main()
        assert execute.call_count == 2
        receipt["image_prefetch"] = not prefetch
        (probe / "PROBE_COMPLETED.json").write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="GPU probe does not match"):
            train_best_mdd_posttraining.main()
        assert execute.call_count == 2
