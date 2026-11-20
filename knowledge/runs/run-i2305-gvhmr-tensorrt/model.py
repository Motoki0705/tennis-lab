"""SIGA24 denoiser bindings and its variable-length attention contract."""

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from src.submodules.configuration import SmplAssetPaths
from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE
from src.submodules.vendor.gvhmr.utils.net_utils import length_to_mask

WINDOW = 120
ROPE_LIMIT = 4096
CHECKPOINT = "gvhmr/gvhmr_siga24_release.ckpt"
PLAN = "gvhmr.plan"
INPUT_WIDTHS = {
    "length": (),
    "obs": (17, 3),
    "f_cliffcam": (3,),
    "f_cam_angvel": (6,),
    "f_imgseq": (1024,),
}
OUTPUT_WIDTHS = {
    "pred_x": (151,),
    "pred_cam": (3,),
    "static_conf_logits": (6,),
    "mask_out": (),
    "length_out": (),
}


def input_shapes(frames, batch_size=1):
    return {
        key: (batch_size,) if key == "length" else (batch_size, frames, *tail)
        for key, tail in INPUT_WIDTHS.items()
    }


def output_shapes(frames, batch_size=1):
    return {
        key: (batch_size,) if key == "length_out" else (batch_size, frames, *tail)
        for key, tail in OUTPUT_WIDTHS.items()
    }


def pack_sequences(sequences):
    """Combine whole, equal-length person sequences along the person axis."""
    frames = int(sequences[0]["length"][0])
    shapes = input_shapes(frames)
    if any(
        arrays["length"].tolist() != [frames]
        or any(arrays[name].shape != shape for name, shape in shapes.items())
        for arrays in sequences
    ):
        raise ValueError("All GVHMR sequences must have the same complete frame length")
    return {
        name: np.concatenate([arrays[name] for arrays in sequences], axis=0)
        for name in shapes
    }


def smpl_assets(root):
    aux = root / "body_models/gvhmr"
    return SmplAssetPaths(
        hmr2_mean_params=aux / "smpl_mean_params.npz",
        smplx_to_smpl=aux / "smplx2smpl_sparse.pt",
        smpl_coco17_regressor=aux / "smpl_coco17_J_regressor.pt",
        smplx_verts437=aux / "smplx_verts437.pt",
        smpl_neutral_joint_regressor=aux / "smpl_neutral_J_regressor.pt",
    )


def load_network(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    prefix = "pipeline.denoiser3d."
    model = NetworkEncoderRoPE().eval()
    model.load_state_dict(
        {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        },
        strict=True,
    )
    if model.max_len != WINDOW or any(
        block.attn.rope.max_seq_len != ROPE_LIMIT for block in model.blocks
    ):
        raise ValueError("The released attention contract changed")
    return DenoiserIO(model)


def attention_mask(length, device):
    """Vectorized equivalent of the released per-row clipping rules, including L <= 120."""
    positions = torch.arange(length, device=device)
    last_start = (positions[-1] + 1 - WINDOW).clamp_min(0)
    starts = torch.minimum((positions - WINDOW // 2).clamp_min(0), last_start)
    ends = (positions + WINDOW // 2).clamp_min(WINDOW)
    return (positions[None, :] < starts[:, None]) | (
        positions[None, :] >= ends[:, None]
    )


class DenoiserIO(nn.Module):
    """Tensor-only boundary around the unchanged eager denoiser."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, length, obs, f_cliffcam, f_cam_angvel, f_imgseq):
        values = self.network(length, obs, f_cliffcam, f_cam_angvel, f_imgseq)
        mask = length_to_mask(length, obs.shape[1])
        return (
            values["pred_x"],
            values["pred_cam"],
            values["static_conf_logits"],
            mask,
            mask.sum(dim=1),
        )


@contextmanager
def dynamic_attention(wrapper):
    """Keep vendor layers and replace the export's length-dependent Python loop."""
    network = wrapper.network
    window = network.max_len
    mask = None

    def set_mask(_model, args):
        nonlocal mask
        obs = args[1]
        mask = attention_mask(obs.shape[1], obs.device)

    def use_mask(_block, args, kwargs):
        return args, {**kwargs, "attn_mask": mask}

    network.max_len = ROPE_LIMIT
    handles = [network.register_forward_pre_hook(set_mask)]
    handles.extend(
        block.register_forward_pre_hook(use_mask, with_kwargs=True)
        for block in network.blocks
    )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        network.max_len = window
