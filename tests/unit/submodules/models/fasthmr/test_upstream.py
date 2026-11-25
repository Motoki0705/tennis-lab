"""The upstream initialization boundary is exercised with small CPU-only models."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest
import torch

import src.submodules.models.fasthmr.upstream as upstream
from src.submodules.models.fasthmr.assets import FastHMRConfig
from src.submodules.models.fasthmr.inference import FastHMR


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_vae_restores_on_cpu_before_moving_to_configured_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str,
) -> None:
    cfg = FastHMRConfig(tmp_path, tmp_path, tmp_path / "coco", device, "pt32", 1, 0)
    cfg.demo_config.parent.mkdir()
    cfg.demo_config.write_text("model_type: videodiffhmr\n")
    weights = cfg.inference_weights
    placements: list[tuple[str, str, float]] = []

    class Model:
        def load_state_dict(self, state: dict[str, Any], *, strict: bool) -> None:
            assert strict

        def eval(self) -> Self:
            return self

        def to(self, destination: str) -> Self:
            return self

    class Block:
        pass

    class Camera(Model):
        def __init__(self, num_layers: int) -> None:
            self.backbone = SimpleNamespace(blocks=[Block() for _ in range(num_layers)])

    class Diffusion(Model):
        def __init__(self, vae_ckpt_path: str) -> None:
            self.vae = torch.nn.Linear(1, 1, bias=False)
            self.load_vae_weights(vae_ckpt_path)

        def load_vae_weights(self, checkpoint_path: str) -> None:
            # Mirrors the pinned constructor's unqualified torch.load call.
            self.vae.load_state_dict(torch.load(checkpoint_path)["model"], strict=True)

        def to(self, destination: str) -> Self:
            placements.append((str(self.vae.weight.device), destination, self.vae.weight.item()))
            return self

    def load_checkpoint(path: Path | str, *, map_location: str | None = None,
                        weights_only: bool | None = None) -> dict[str, Any]:
        if Path(path) == weights["vae"]:
            if map_location != "cpu":
                raise RuntimeError("cannot restore CUDA-tagged VAE checkpoint without CUDA")
            assert weights_only is True
            return {"model": {"weight": torch.tensor([[3.25]])}}
        assert map_location == "cpu"
        return {"n_blocks": 1, "weights": {}, "state_dict": {}, "model": {}}

    modules = {
        "data.preprocess.camerahmr.camerahmr_model": SimpleNamespace(CameraHMR=Camera),
        "data.preprocess.camerahmr.heads.smpl_head_cliff": SimpleNamespace(SMPL_MEAN_PARAMS_FILE="mean.npz"),
        "data.preprocess.camerahmr.cam_model.fl_net": SimpleNamespace(FLNet=Model),
        "model.hmr2_tools.patch_vit": SimpleNamespace(ToMeBlock=Block, apply_patch=lambda _: None),
        "model.fasthmr": SimpleNamespace(FastHMR=Diffusion),
        "data.preprocess.camerahmr.datasets.utils": SimpleNamespace(generate_image_patch_cv2=object()),
    }
    monkeypatch.setattr(upstream, "import_module", modules.__getitem__)
    monkeypatch.setattr(upstream, "_official_imports", lambda _: nullcontext())
    monkeypatch.setattr(upstream, "_require_pinned_asset", lambda *_: None)
    monkeypatch.setattr(torch, "load", load_checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: path))

    model = FastHMR(cfg)
    model.load()
    assert model.models is not None and isinstance(model.models[2], Diffusion)
    assert placements == [("cpu", device, 3.25)]
