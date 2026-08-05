"""GVHMR inference pipeline (adapted from hmr4d.model.gvhmr).

Combines ``gvhmr_pipeline.Pipeline`` (inference branch only) and the DemoPL
wrapper. Hydra/MainStore instantiation is replaced by explicit factories; the
training losses are not vendored.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.submodules.vendor.gvhmr import stats_compose
from src.submodules.vendor.gvhmr.endecoder import EnDecoder
from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE
from src.submodules.vendor.gvhmr.postprocess import (
    pp_static_joint,
    pp_static_joint_cam,
    process_ik,
)
from src.submodules.vendor.gvhmr.utils.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    normalize_kp2d,
)
from src.submodules.vendor.gvhmr.utils.hmr_global import (
    get_tgtcoord_rootparam,
    rollout_local_transl_vel,
)
from src.submodules.vendor.gvhmr.utils.net_utils import gaussian_smooth
from src.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

_LOGGER = logging.getLogger(__name__)

# EnDecoder statistics used by the released SIGA24 demo checkpoint
DEMO_ENDECODER_STATS = "MM_V1_AMASS_LOCAL_BEDLAM_CAM"


class Pipeline(nn.Module):
    """Inference-only GVHMR pipeline (denoiser3d + endecoder + postproc)."""

    def __init__(self, denoiser3d: nn.Module, endecoder: EnDecoder, normalize_cam_angvel: bool = True):
        super().__init__()
        self.denoiser3d = denoiser3d
        self.endecoder = endecoder
        self.normalize_cam_angvel = normalize_cam_angvel
        if normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer("cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False)
            self.register_buffer("cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False)

    def forward(self, inputs, postproc=True, *, static_cam):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample

        # *. Conditions
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        f_cam_angvel = inputs["cam_angvel"]
        if self.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        f_condition = {
            "obs": inputs["obs"],  # (B, L, J, 3)
            "f_cliffcam": cliff_cam,  # (B, L, 3)
            "f_cam_angvel": f_cam_angvel,  # (B, L, C=6)
            "f_imgseq": inputs["f_imgseq"],  # (B, L, C=1024)
        }

        # Forward & output
        model_output = self.denoiser3d(length=length, **f_condition)  # pred_x, pred_cam, static_conf_logits
        decode_dict = self.endecoder.decode(model_output["pred_x"])  # (B, L, C) -> dict
        outputs.update({"model_output": model_output, "decode_dict": decode_dict})

        # Post-processing
        outputs["pred_smpl_params_incam"] = {
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "betas": decode_dict["betas"],  # (B, L, 10)
            "global_orient": decode_dict["global_orient"],  # (B, L, 3)
            "transl": compute_transl_full_cam(model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]),
        }
        pred_smpl_params_global = get_smpl_params_w_Rt_v2(  # This function has for-loop
            global_orient_gv=decode_dict["global_orient_gv"],
            local_transl_vel=decode_dict["local_transl_vel"],
            global_orient_c=decode_dict["global_orient"],
            cam_angvel=inputs["cam_angvel"],
        )
        outputs["pred_smpl_params_global"] = {
            "body_pose": decode_dict["body_pose"],
            "betas": decode_dict["betas"],
            **pred_smpl_params_global,
        }
        outputs["static_conf_logits"] = model_output["static_conf_logits"]

        if postproc:  # apply post-processing
            if static_cam:  # extra post-processing to utilize static camera prior
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(outputs, self.endecoder)
            else:
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.endecoder)
            body_pose = process_ik(outputs, self.endecoder)
            decode_dict["body_pose"] = body_pose
            outputs["pred_smpl_params_global"]["body_pose"] = body_pose
            outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

        return outputs


@autocast("cuda", enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0))
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)  # Smooth
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt


class GvhmrDemoModel(nn.Module):
    """Checkpoint-compatible replacement for hmr4d's ``DemoPL``.

    The submodule is named ``pipeline`` so released checkpoints
    (``pipeline.denoiser3d.*`` keys) load directly.
    """

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline

    @torch.no_grad()
    def predict(self, data, *, static_cam):
        """Run GVHMR on a single-person sequence (batch dim is added).

        data: {
            "length": scalar Tensor,
            "kp2d": (F, 17, 3),
            "bbx_xys": (F, 3),
            "K_fullimg": (F, 3, 3),
            "cam_angvel": (F, 6),
            "f_imgseq": (F, 1024),
        }
        """
        device = next(self.parameters()).device
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    def load_pretrained_model(self, ckpt_path):
        """Load a released GVHMR checkpoint (pytorch-lightning state_dict)."""
        _LOGGER.info("Loading GVHMR checkpoint: %s", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            _LOGGER.warning("Missing keys: %s", missing)
        if len(unexpected) > 0:
            _LOGGER.warning("Unexpected keys: %s", unexpected)


def build_gvhmr_demo_model(
    checkpoint_path,
    body_models_dir,
    *,
    bundled_assets,
) -> GvhmrDemoModel:
    """Build the GVHMR demo model with the released SIGA24 configuration."""
    denoiser3d = NetworkEncoderRoPE()
    endecoder = EnDecoder(
        body_model_path=body_models_dir,
        bundled_assets=bundled_assets,
        stats_name=DEMO_ENDECODER_STATS,
    )
    model = GvhmrDemoModel(Pipeline(denoiser3d, endecoder, normalize_cam_angvel=True))
    model.load_pretrained_model(checkpoint_path)
    return model
