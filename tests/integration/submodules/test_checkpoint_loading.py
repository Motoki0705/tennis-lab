"""Integration: released checkpoints load strictly into the vendored networks.

These validate that the vendored architectures are state-dict compatible with
the checkpoints under ckpt/ (symlinks to third_party/GVHMR). They require the
local checkpoint files and a few GB of RAM, hence the markers.
"""

from pathlib import Path

import pytest
import torch

from src.utils.paths import PROJECT_ROOT

VITPOSE_CKPT = PROJECT_ROOT / "ckpt/vitpose/vitpose-h-multi-coco.pth"
HMR2_CKPT = PROJECT_ROOT / "ckpt/hmr2/epoch=10-step=25000.ckpt"
GVHMR_CKPT = PROJECT_ROOT / "ckpt/gvhmr/gvhmr_siga24_release.ckpt"
HMR2_MEAN_PARAMS = (
    PROJECT_ROOT / "src/submodules/vendor/gvhmr/hmr2/smpl_mean_params.npz"
)


def _needs(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"missing local checkpoint: {path}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.local_data
class TestCheckpointCompatibility:
    @_needs(VITPOSE_CKPT)
    def test_vitpose_strict_load_and_forward(self):
        from src.submodules.vendor.gvhmr.vitpose import build_vitpose_huge
        from src.submodules.vendor.gvhmr.vitpose.heatmap_head import ViTPoseHeadConfig

        model = build_vitpose_huge(
            str(VITPOSE_CKPT),
            head_config=ViTPoseHeadConfig(
                in_channels=1280,
                out_channels=17,
                num_deconv_layers=2,
                num_deconv_filters=(256, 256),
                num_deconv_kernels=(4, 4),
                final_conv_kernel=1,
                num_conv_layers=0,
                num_conv_kernels=(),
            ),
        ).eval()
        with torch.no_grad():
            heatmap = model(torch.rand(1, 3, 256, 192))
        assert heatmap.shape == (1, 17, 64, 48)

    @_needs(HMR2_CKPT)
    def test_hmr2_strict_load_and_forward(self):
        from src.submodules.vendor.gvhmr.hmr2 import load_hmr2

        model = load_hmr2(str(HMR2_CKPT), mean_params_path=HMR2_MEAN_PARAMS).eval()
        with torch.no_grad():
            features = model({"img": torch.rand(1, 3, 256, 256)})
        assert features.shape == (1, 1024)

    @_needs(GVHMR_CKPT)
    def test_gvhmr_denoiser_keys_match(self):
        from src.submodules.vendor.gvhmr.network.relative_transformer import (
            NetworkEncoderRoPE,
        )

        state_dict = torch.load(GVHMR_CKPT, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        prefix = "pipeline.denoiser3d."
        stripped = {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        assert stripped, "checkpoint has no pipeline.denoiser3d.* keys"

        net = NetworkEncoderRoPE()
        missing, unexpected = net.load_state_dict(stripped, strict=False)
        assert missing == []
        assert unexpected == []
