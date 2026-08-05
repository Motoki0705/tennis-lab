"""SMPL transformer decoder head (adapted from hmr4d.network.hmr2.smpl_head).

The yacs config dependency is replaced with explicit constructor defaults that
match HMR2.0a's ``model_config.yaml``.
"""

import einops
import numpy as np
import torch
import torch.nn as nn

from .geometry import rot6d_to_rotmat
from .pose_transformer import TransformerDecoder


class SMPLTransformerDecoderHead(nn.Module):
    """Cross-attention based SMPL Transformer decoder."""

    def __init__(
        self,
        num_body_joints=23,
        context_dim=1280,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        *,
        mean_params_path,
    ):
        super().__init__()
        self.joint_rep_type = "6d"
        self.joint_rep_dim = 6
        npose = self.joint_rep_dim * (num_body_joints + 1)
        self.npose = npose
        self.num_body_joints = num_body_joints
        self.input_is_mean_shape = False
        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=context_dim,
        )
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load(mean_params_path)
        init_body_pose = torch.from_numpy(
            mean_params["pose"].astype(np.float32)
        ).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(
            0
        )
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, only_return_token_out=False):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []

        # Input token to transformer is zero token
        token = torch.zeros(batch_size, 1, 1).to(x.device)

        # Pass through transformer
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1)  # (B, C)

        if only_return_token_out:
            return token_out

        # Readout from token_out
        pred_body_pose = self.decpose(token_out) + pred_body_pose
        pred_betas = self.decshape(token_out) + pred_betas
        pred_cam = self.deccam(token_out) + pred_cam
        pred_body_pose_list.append(pred_body_pose)
        pred_betas_list.append(pred_betas)
        pred_cam_list.append(pred_cam)

        pred_smpl_params_list = {}
        pred_smpl_params_list["body_pose"] = torch.cat(
            [
                rot6d_to_rotmat(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :]
                for pbp in pred_body_pose_list
            ],
            dim=0,
        )
        pred_smpl_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = rot6d_to_rotmat(pred_body_pose).view(
            batch_size, self.num_body_joints + 1, 3, 3
        )

        pred_smpl_params = {
            "global_orient": pred_body_pose[:, [0]],
            "body_pose": pred_body_pose[:, 1:],
            "betas": pred_betas,
        }
        return pred_smpl_params, pred_cam, pred_smpl_params_list, token_out
