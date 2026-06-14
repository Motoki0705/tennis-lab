"""Contract / smoke tests for the DINOv3 tennis SSL task.

These tests assert the three acceptance conditions of issue #498 without needing
network access or the 342 MB pretrained checkpoint:

    * the data-collection pipeline produces a manifest-backed image folder,
    * LoRA adapters (and only the adapters/heads) receive gradients,
    * the DINOv3 self-distillation strategy is functioning: the EMA teacher
      tracks the student and the combined DINO + iBOT + KoLeo loss decreases
      when overfitting a fixed batch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

# Small but real DINOv3 ViT-B geometry: 112px global crops -> 7x7 = 49 patches.
GLOBAL_SIZE = 112
LOCAL_SIZE = 48
PATCH_SIZE = 16
NUM_PATCHES = (GLOBAL_SIZE // PATCH_SIZE) ** 2


def _smoke_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "backbone_name": "dinov3_vitb16",
                "checkpoint_path": None,
                "load_pretrained": False,  # keep the test offline and light
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "bias": "none",
                    "target_modules": ["qkv", "proj", "fc1", "fc2"],
                },
                "head": {"hidden_dim": 256, "bottleneck_dim": 64, "nlayers": 2},
                "dino": {"out_dim": 1024},
                "ibot": {"enabled": True, "out_dim": 1024},
            },
            "training": {
                "loss": {
                    "student_temp": 0.1,
                    "center_momentum": 0.9,
                    "dino_weight": 1.0,
                    "ibot_weight": 1.0,
                    "koleo_weight": 0.1,
                },
                "schedule": {
                    "teacher_temp": 0.07,
                    "teacher_temp_warmup": 0.04,
                    "teacher_temp_warmup_epochs": 1,
                    "momentum_base": 0.9,
                    "momentum_final": 1.0,
                },
                "optimizer": {
                    "lr": 1.0e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "warmup_steps": 0,
                    "min_lr_ratio": 1.0,
                },
                "trainer": {"max_epochs": 1},
            },
        }
    )


def _fake_batch(batch_size: int = 2) -> dict:
    torch.manual_seed(0)
    return {
        "global_crops": [
            torch.randn(batch_size, 3, GLOBAL_SIZE, GLOBAL_SIZE) for _ in range(2)
        ],
        "local_crops": [
            torch.randn(batch_size, 3, LOCAL_SIZE, LOCAL_SIZE) for _ in range(2)
        ],
        "masks": [torch.rand(batch_size, NUM_PATCHES) < 0.4 for _ in range(2)],
    }


@pytest.mark.integration
def test_collection_pipeline_writes_manifest(tmp_path: Path) -> None:
    """The collector produces an image folder + manifest (synthetic, offline)."""
    output_dir = tmp_path / "tennis_smoke"
    command = [
        sys.executable,
        "-m",
        "src.tasks.dino_ssl.scripts.collect",
        f"collector.output_dir={output_dir}",
        "collector.sources=[]",
        "collector.min_images=6",
        "collector.synthetic_fallback.enabled=true",
        "collector.synthetic_fallback.size=64",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    from src.tasks.dino_ssl.generate_dataset.manifest import read_manifest

    manifest = read_manifest(output_dir)
    assert manifest.num_images == 6
    assert all(path.is_file() for path in manifest.image_paths())


@pytest.mark.slow
@pytest.mark.integration
def test_lora_self_distillation_trains() -> None:
    """LoRA adapters update, the EMA teacher tracks the student, loss decreases."""
    from src.tasks.dino_ssl.models.backbone import count_trainable_parameters
    from src.tasks.dino_ssl.training.lightning_module import DinoSSLLightningModule

    torch.manual_seed(0)
    module = DinoSSLLightningModule(_smoke_config(), steps_per_epoch=8)
    module.train()

    # LoRA-only: the frozen backbone dominates the parameter count.
    trainable, total = count_trainable_parameters(module.network.student)
    assert 0 < trainable < total

    optimizer = module.configure_optimizers()["optimizer"]
    batch = _fake_batch()

    # Snapshot a teacher parameter to confirm the EMA update moves it.
    teacher_param = next(module.network.teacher.parameters())
    teacher_before = teacher_param.detach().clone()

    losses: list[float] = []
    saw_lora_grad = False
    for _ in range(6):
        optimizer.zero_grad()
        out = module._compute_losses(batch, update_center=True)
        loss = out["total"]
        assert torch.isfinite(loss), "SSL loss must stay finite"
        loss.backward()

        for name, param in module.network.student.named_parameters():
            if "lora" in name and param.requires_grad and param.grad is not None:
                saw_lora_grad = True
                break

        optimizer.step()
        module.network.update_teacher(momentum=0.9)
        losses.append(float(loss.detach()))

    assert saw_lora_grad, "LoRA adapters received no gradient"
    assert not torch.equal(teacher_before, teacher_param.detach()), (
        "EMA teacher did not update"
    )
    # Overfitting a fixed batch should reduce the self-distillation loss.
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
