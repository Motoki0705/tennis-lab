"""GVHMR-style monocular PLCS 3D skeleton + HTD-Refine Stage A (PVA-Heads) model.

Architecture follows GVHMR's ``NetworkEncoderRoPE`` (``hidden_dim=512``,
``num_layers=12``, ``num_heads=8``, ``mlp_ratio=4.0``, RoPE attention, AdaLN-Zero
residual gates, zero-initialized Early Fusion condition embedders) while omitting
HMR2 image features completely:

- Inputs: bbox-normalized COCO-17 2D keypoints ``kp2d_norm (B, T, 17, 3)``,
  CLIFF bbox token ``cliffcam (B, T, 3)``, 21-D camera intrinsics/extrinsics
  token ``cam_feat (B, T, 21)``, ``bbx_xys (B, T, 3)``, ``frame_observed (B, T)``,
  ``padding_mask (B, T)``, and calibrated camera parameters ``camera_*_in``.
- Outputs: physical court-space 3D COCO-17 skeleton ``world_kp_3d (B, T, 17, 3)`` [m],
  court-normalized ``position (B, T, 3)``, ``rotation (B, T, 2)``,
  ``canonical_pose (B, T, 17, 3)``, CLIFF ``pred_cam (B, T, 3)``, and optional
  HTD-Refine Stage A ``pred_vel_3d``, ``pred_acc_3d``, ``static_contact_logits``.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.models.vision_transformer import Mlp
from torch import Tensor

from src.submodules.vendor.gvhmr.network.layer import zero_module
from src.submodules.vendor.gvhmr.network.rotary_embedding import ROPE
from src.tasks.plcs.geometry.monocular_camera_bbox import (
    BBX_XYS_DIM,
    CAMERA_FEATURE_DIM,
    CLIFFCAM_DIM,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    center_local_pose_on_hips,
    cliff_and_local_pose_to_world_coco17,
)
from src.utils.geometry.court_pose import world_pose_to_canonical_pose
from src.utils.schema.court_normalization import normalize_court_position
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from src.tasks.plcs.configuration import PLCSModelConfig

STATIC_CONTACT_JOINT_INDICES: Final[tuple[int, int, int, int]] = (15, 16, 13, 14)
NUM_STATIC_CONTACT_JOINTS: Final[int] = len(STATIC_CONTACT_JOINT_INDICES)

# Non-degenerate athletic ready-stance COCO-17 template in camera coordinates [m]
# (+X right, +Y down, +Z forward/depth). Flexed elbows and knees ensure every
# bone length, interior angle, limb dihedral torsion plane, and torso twist plane
# strictly satisfies ``SkeletonValues.require_valid`` from step 0.
_CANONICAL_CAMERA_COCO17_TEMPLATE: Final[tuple[tuple[float, float, float], ...]] = (
    (0.00, -0.64, -0.06),   # 0 nose
    (-0.03, -0.67, -0.07),  # 1 left_eye
    (0.03, -0.67, -0.07),   # 2 right_eye
    (-0.07, -0.65, -0.02),  # 3 left_ear
    (0.07, -0.65, -0.02),   # 4 right_ear
    (-0.19, -0.48, 0.00),   # 5 left_shoulder
    (0.19, -0.48, 0.00),    # 6 right_shoulder
    (-0.28, -0.24, -0.08),  # 7 left_elbow
    (0.28, -0.24, -0.08),   # 8 right_elbow
    (-0.22, -0.08, -0.24),  # 9 left_wrist
    (0.22, -0.08, -0.24),   # 10 right_wrist
    (-0.12, 0.00, 0.00),    # 11 left_hip
    (0.12, 0.00, 0.00),     # 12 right_hip
    (-0.14, 0.42, -0.08),   # 13 left_knee
    (0.14, 0.42, -0.08),    # 14 right_knee
    (-0.14, 0.84, 0.04),    # 15 left_ankle
    (0.14, 0.84, 0.04),     # 16 right_ankle
)


def _approx_gelu() -> nn.Module:
    return nn.GELU(approximate="tanh")


def _small_init_linear(
    in_features: int, out_features: int, std: float = 1e-3
) -> nn.Linear:
    """Near-zero Linear readout that preserves step-1 gradient flow into the trunk."""
    layer = nn.Linear(in_features, out_features)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)
    return layer


def build_monocular_sliding_padding_mask(
    padding_mask: Tensor,
    *,
    sliding_window_len: int,
    frame_observed: Tensor | None = None,
) -> Tensor:
    """Build NaN-safe, segment-isolated ``(B, 1, L, L)`` attention mask (``True`` = masked).

    - Active frames are ``~padding_mask`` (and ``frame_observed`` when provided).
    - Active queries ``i`` attend strictly to active keys ``j`` within the same
      contiguous active segment (never crossing an internal unobserved or padded
      gap) and within ``|i - j| <= sliding_window_len // 2``.
    - Inactive queries ``i`` attend solely to ``j == i`` so ``softmax`` never
      encounters an all-``-inf`` row (avoiding ``NaN`` forward/backward values).
    """
    if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
        raise ValueError(
            f"padding_mask must be bool of shape (B, L), got {tuple(padding_mask.shape)}."
        )
    if frame_observed is not None:
        if frame_observed.shape != padding_mask.shape or frame_observed.dtype != torch.bool:
            raise ValueError(
                "frame_observed must be bool of shape (B, L) matching padding_mask."
            )
        inactive = padding_mask | (~frame_observed)
    else:
        inactive = padding_mask

    _, seq_len = padding_mask.shape
    device = padding_mask.device
    idx = torch.arange(seq_len, device=device)
    half_win = max(sliding_window_len // 2, 0)
    outside_window = (idx[:, None] - idx[None, :]).abs() > half_win  # (L, L)

    # Contiguous active segments share the same cumulative count of inactive barriers.
    seg_id = torch.cumsum(inactive.to(dtype=torch.int64), dim=1)  # (B, L)
    different_segment = seg_id[:, :, None] != seg_id[:, None, :]  # (B, L, L)

    # Mask any key outside the local window, in another segment, or inactive.
    masked = outside_window.unsqueeze(0) | different_segment | inactive[:, None, :]

    # For inactive queries, mask all keys EXCEPT self-diagonal (i, i) so softmax is finite.
    eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)  # (1, L, L)
    inactive_query_row = ~eye
    masked = torch.where(inactive[:, :, None], inactive_query_row, masked)
    return masked.unsqueeze(1)  # (B, 1, L, L)


class _MonocularRoPEAttention(nn.Module):
    """GVHMR-compatible RoPE self-attention accepting a ``(B, 1, L, L)`` mask."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.rope = ROPE(self.head_dim, max_seq_len=max_seq_len)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, attn_mask_4d: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        xq = self.query(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.key(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xv = self.value(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)
        xk = self.rope.rotate_queries_or_keys(xk)

        attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(
            self.head_dim
        )
        attn_score = attn_score.masked_fill(attn_mask_4d, float("-inf"))
        attn_prob = self.dropout(torch.softmax(attn_score, dim=-1))
        out = einsum(attn_prob, xv, "b n i j, b n j c -> b n i c")
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        return self.proj(out)


class MonocularEncoderRoPEBlock(nn.Module):
    """GVHMR ``EncoderRoPEBlock`` (Pre-LN + RoPE-MHA + AdaLN-Zero residual gates)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = _MonocularRoPEAttention(
            hidden_size, num_heads, dropout=dropout, max_seq_len=max_seq_len
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=_approx_gelu,
            drop=dropout,
        )
        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x: Tensor, *, attn_mask_4d: Tensor, valid_3d: Tensor) -> Tensor:
        x = x + self.gate_msa * self.attn(self.norm1(x), attn_mask_4d)
        x = x * valid_3d
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x * valid_3d


class PLCSMonocularGVHMRModel(nn.Module):
    """GVHMR-style Early Fusion monocular PLCS 3D skeleton + PVA-Heads estimator."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        sliding_window_len: int = 120,
        enable_pva_heads: bool = True,
        predict_static_contact: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be positive and divisible by num_heads={num_heads}."
            )
        if (hidden_dim // num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim (hidden_dim // num_heads).")
        if num_layers < 0 or max_seq_len <= 0 or sliding_window_len <= 0:
            raise ValueError("num_layers >= 0, max_seq_len > 0, sliding_window_len > 0 required.")
        if predict_static_contact and not enable_pva_heads:
            raise ValueError("predict_static_contact=True requires enable_pva_heads=True.")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.max_seq_len = int(max_seq_len)
        self.sliding_window_len = int(sliding_window_len)
        self.enable_pva_heads = bool(enable_pva_heads)
        self.predict_static_contact = bool(predict_static_contact)

        # 1. GVHMR 2D keypoint token encoder (17 joints * 32-D -> 2*D -> D)
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(NUM_HUMAN_KP, 32) * 0.02)
        self.embed_noisyobs = Mlp(
            in_features=NUM_HUMAN_KP * 32,
            hidden_features=self.hidden_dim * 2,
            out_features=self.hidden_dim,
            drop=self.dropout,
        )
        self.unobserved_frame_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # 2. Zero-initialized Early Fusion condition embedders (CLIFF + Camera; NO HMR2)
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(CLIFFCAM_DIM, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            zero_module(nn.Linear(self.hidden_dim, self.hidden_dim)),
        )
        self.cam_feat_embedder = nn.Sequential(
            nn.Linear(CAMERA_FEATURE_DIM, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            zero_module(nn.Linear(self.hidden_dim, self.hidden_dim)),
        )

        # 3. RoPE Relative Transformer backbone
        self.blocks = nn.ModuleList(
            [
                MonocularEncoderRoPEBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    max_seq_len=self.max_seq_len,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        # 4. CLIFF weak-perspective root + hip-centered camera-local COCO-17 pose heads
        self.pred_cam_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            _small_init_linear(self.hidden_dim, 3),
        )
        # Initial scale bias so softplus(0 + bias) + 0.25 == 1.0606 (GVHMR mean scale).
        init_s_excess = 1.0606 - 0.25
        init_s_bias = math.log(math.expm1(init_s_excess))
        self.register_buffer(
            "_pred_cam_init_bias",
            torch.tensor([init_s_bias, 0.0, 0.0], dtype=torch.float32),
            persistent=False,
        )

        template = torch.tensor(_CANONICAL_CAMERA_COCO17_TEMPLATE, dtype=torch.float32)
        template = center_local_pose_on_hips(template)
        self.register_buffer("template_local_pose_cam", template, persistent=False)

        self.local_pose_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            _small_init_linear(self.hidden_dim, NUM_HUMAN_KP * 3),
        )
        # Camera-local pelvis offset & transverse yaw refinement (equivariant under world rigid transforms!)
        self.pelvis_offset_cam_head = _small_init_linear(self.hidden_dim, 3)
        self.yaw_delta_head = _small_init_linear(self.hidden_dim, 2)

        # 5. HTD-Refine Stage A: PVA-Heads (Position-Velocity-Acceleration + Static Contact)
        if self.enable_pva_heads:
            self.pva_conv = nn.Conv1d(
                self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1
            )
            self.pva_act = nn.SiLU()
            self.pva_block = MonocularEncoderRoPEBlock(
                self.hidden_dim,
                self.num_heads,
                mlp_ratio=2.0,
                dropout=self.dropout,
                max_seq_len=self.max_seq_len,
            )
            self.pva_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
            self.vel_cam_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                _small_init_linear(self.hidden_dim, NUM_HUMAN_KP * 3),
            )
            self.acc_cam_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                _small_init_linear(self.hidden_dim, NUM_HUMAN_KP * 3),
            )
            if self.predict_static_contact:
                self.static_contact_head = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.SiLU(),
                    _small_init_linear(
                        self.hidden_dim // 2, NUM_STATIC_CONTACT_JOINTS
                    ),
                )

    @classmethod
    def from_config(cls, config: PLCSModelConfig) -> PLCSMonocularGVHMRModel:
        """Instantiate from validated ``PLCSModelConfig``."""
        return cls(
            hidden_dim=config.integer("hidden_dim"),
            num_layers=config.integer("num_layers"),
            num_heads=config.integer("num_heads"),
            mlp_ratio=config.number("mlp_ratio"),
            dropout=config.number("dropout"),
            max_seq_len=config.integer("max_seq_len"),
            sliding_window_len=config.integer("sliding_window_len"),
            enable_pva_heads=config.boolean("enable_pva_heads"),
            predict_static_contact=config.boolean("predict_static_contact"),
        )

    @staticmethod
    def _expand_cam_time(
        param: Tensor, *, bsz: int, seq_len: int, trailing_dims: int
    ) -> Tensor:
        lead = param.shape if trailing_dims == 0 else param.shape[:-trailing_dims]
        tail = () if trailing_dims == 0 else param.shape[-trailing_dims:]
        if lead == (bsz, seq_len):
            return param
        if lead == (bsz,):
            if trailing_dims == 0:
                return param.unsqueeze(-1).expand(bsz, seq_len)
            return param.reshape(bsz, 1, *tail).expand(bsz, seq_len, *tail)
        raise ValueError(
            f"Camera tensor leading shape {tuple(lead)} must be ({bsz},) or ({bsz}, {seq_len})."
        )

    def forward(
        self,
        *,
        kp2d_norm: Tensor,
        cliffcam: Tensor,
        cam_feat: Tensor,
        bbx_xys: Tensor,
        frame_observed: Tensor,
        padding_mask: Tensor,
        camera_R_in: Tensor,
        camera_C_in: Tensor,
        camera_f_in: Tensor,
        camera_cx_in: Tensor,
        camera_cy_in: Tensor,
        camera_fy_in: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run monocular Early Fusion forward pass and lift to physical court 3D [m]."""
        bsz, seq_len, n_joints, kp_dim = kp2d_norm.shape
        if n_joints != NUM_HUMAN_KP or kp_dim != 3:
            raise ValueError(
                f"kp2d_norm must have shape (B, T, 17, 3), got {tuple(kp2d_norm.shape)}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if bbx_xys.shape != (bsz, seq_len, BBX_XYS_DIM):
            raise ValueError("bbx_xys must have shape (B, T, 3).")
        if frame_observed.shape != (bsz, seq_len) or padding_mask.shape != (bsz, seq_len):
            raise ValueError("frame_observed and padding_mask must have shape (B, T).")

        fy_in = camera_f_in if camera_fy_in is None else camera_fy_in
        valid_frame = ~padding_mask
        obs_frame = frame_observed & valid_frame
        active_3d = obs_frame.unsqueeze(-1).to(dtype=kp2d_norm.dtype)

        # 1. Encode 2D COCO-17 keypoints with learned occlusion tokens
        joint_vis = (kp2d_norm[..., 2:3] > 0.0) & obs_frame[:, :, None, None]
        kp_xy = torch.where(joint_vis, kp2d_norm[..., :2], torch.zeros_like(kp2d_norm[..., :2]))
        f_obs_vis = self.learned_pos_linear(kp_xy)
        f_obs_inv = self.learned_pos_params.view(1, 1, NUM_HUMAN_KP, 32).expand_as(f_obs_vis)
        f_obs = torch.where(joint_vis, f_obs_vis, f_obs_inv)
        x = self.embed_noisyobs(f_obs.reshape(bsz, seq_len, NUM_HUMAN_KP * 32))

        unobs_valid = valid_frame & (~obs_frame)
        x = x + unobs_valid.unsqueeze(-1).to(dtype=x.dtype) * self.unobserved_frame_embed

        # 2. Early Fusion of CLIFF bbox + 21-D camera calibration features
        cliff_clean = torch.where(
            obs_frame.unsqueeze(-1), cliffcam, torch.zeros_like(cliffcam)
        )
        cam_clean = torch.where(obs_frame.unsqueeze(-1), cam_feat, torch.zeros_like(cam_feat))
        x = x + self.cliffcam_embedder(cliff_clean) + self.cam_feat_embedder(cam_clean)
        x = x * active_3d

        # 3. RoPE Relative Transformer with NaN-safe, segment-isolated sliding window mask
        attn_mask_4d = build_monocular_sliding_padding_mask(
            padding_mask,
            sliding_window_len=self.sliding_window_len,
            frame_observed=frame_observed,
        )
        for block in self.blocks:
            x = block(x, attn_mask_4d=attn_mask_4d, valid_3d=active_3d)

        x_norm = self.final_norm(x) * active_3d

        # 4. Decode CLIFF root + hip-centered camera-local COCO-17 offsets -> court 3D [m] in FP32
        raw_cam = self.pred_cam_head(x_norm)
        delta_local = self.local_pose_head(x_norm).reshape(
            bsz, seq_len, NUM_HUMAN_KP, 3
        )
        pelvis_delta_cam = self.pelvis_offset_cam_head(x_norm)
        yaw_ab = self.yaw_delta_head(x_norm)

        device_type = x_norm.device.type
        amp_off = (
            torch.amp.autocast(device_type=device_type, enabled=False)
            if device_type in {"cpu", "cuda", "mps", "xpu"}
            else contextlib.nullcontext()
        )
        with amp_off:
            raw_cam_fp32 = raw_cam.float() + self._pred_cam_init_bias.float()
            pred_s = 0.25 + F.softplus(raw_cam_fp32[..., 0])
            pred_tx = raw_cam_fp32[..., 1]
            pred_ty = raw_cam_fp32[..., 2]
            pred_cam = torch.stack([pred_s, pred_tx, pred_ty], dim=-1)

            camera_R_fp32 = camera_R_in.float()
            camera_C_fp32 = camera_C_in.float()
            camera_f_fp32 = camera_f_in.float()
            fy_fp32 = fy_in.float()
            camera_cx_fp32 = camera_cx_in.float()
            camera_cy_fp32 = camera_cy_in.float()
            bbx_fp32 = bbx_xys.float()

            fx_t = self._expand_cam_time(
                camera_f_fp32, bsz=bsz, seq_len=seq_len, trailing_dims=0
            )
            fy_t = self._expand_cam_time(
                fy_fp32, bsz=bsz, seq_len=seq_len, trailing_dims=0
            )
            cx0_t = self._expand_cam_time(
                camera_cx_fp32, bsz=bsz, seq_len=seq_len, trailing_dims=0
            )
            cy0_t = self._expand_cam_time(
                camera_cy_fp32, bsz=bsz, seq_len=seq_len, trailing_dims=0
            )
            R_t = self._expand_cam_time(
                camera_R_fp32, bsz=bsz, seq_len=seq_len, trailing_dims=2
            )

            # On unobserved/padded frames (where detector bbx_xys == 0), use the per-frame
            # principal-point reference box [cx0, cy0, 0.2 * sqrt(fx*fy)] so CLIFF maps
            # [s, tx, ty] -> [tx, ty, 10/s] deterministically without sequence-length coupling.
            f_eff = torch.sqrt(fx_t * fy_t)
            unobs_bbx = torch.stack([cx0_t, cy0_t, 0.2 * f_eff], dim=-1)
            use_det_box = obs_frame & (bbx_fp32[..., 2] > 1e-3)
            bbx_eff = torch.where(use_det_box.unsqueeze(-1), bbx_fp32, unobs_bbx)

            local_pose_cam = center_local_pose_on_hips(
                self.template_local_pose_cam.float().view(1, 1, NUM_HUMAN_KP, 3)
                + delta_local.float()
            )
            world_kp_3d = cliff_and_local_pose_to_world_coco17(
                pred_cam,
                local_pose_cam,
                bbx_eff,
                R_w2c=camera_R_fp32,
                C_w=camera_C_fp32,
                fx=camera_f_fp32,
                fy=fy_fp32,
                cx0=camera_cx_fp32,
                cy0=camera_cy_fp32,
            )

            # Derive court position, court heading (cos, sin), and canonical_pose consistently
            mid_hip_world = 0.5 * (
                world_kp_3d[..., LEFT_HIP_IDX, :] + world_kp_3d[..., RIGHT_HIP_IDX, :]
            )
            pelvis_delta_world = torch.einsum(
                "...ji,...j->...i", R_t, pelvis_delta_cam.float()
            )
            pelvis_world = mid_hip_world + pelvis_delta_world
            position = normalize_court_position(pelvis_world)

            v_trans = (
                world_kp_3d[..., LEFT_HIP_IDX, :2] - world_kp_3d[..., RIGHT_HIP_IDX, :2]
            )
            v_perp = torch.stack([-v_trans[..., 1], v_trans[..., 0]], dim=-1)
            yaw_ab_fp32 = yaw_ab.float()
            heading_raw = (
                (1.0 + yaw_ab_fp32[..., 0:1]) * v_trans
                + yaw_ab_fp32[..., 1:2] * v_perp
            )
            rotation = F.normalize(heading_raw, dim=-1, eps=1e-6)
            canonical_pose = world_pose_to_canonical_pose(world_kp_3d, position, rotation)

        outputs: dict[str, Tensor] = {
            "position": position,
            "rotation": rotation,
            "canonical_pose": canonical_pose,
            "world_kp_3d": world_kp_3d,
            "pred_cam": pred_cam,
        }

        # 5. HTD-Refine Stage A: PVA-Heads (strictly isolated across missing & padded gaps!)
        if self.enable_pva_heads:
            h_pva = (x * active_3d).transpose(1, 2)  # (B, D, T)
            h_pva = self.pva_act(self.pva_conv(h_pva)).transpose(1, 2) * active_3d
            h_pva = self.pva_block(
                x + h_pva, attn_mask_4d=attn_mask_4d, valid_3d=active_3d
            )
            h_pva = self.pva_norm(h_pva) * active_3d

            vel_cam = self.vel_cam_head(h_pva).reshape(bsz, seq_len, NUM_HUMAN_KP, 3)
            acc_cam = self.acc_cam_head(h_pva).reshape(bsz, seq_len, NUM_HUMAN_KP, 3)
            active_3d_fp32 = obs_frame.unsqueeze(-1).to(dtype=torch.float32)
            with amp_off:
                pred_vel_3d = (
                    torch.einsum("...ji,...kj->...ki", R_t, vel_cam.float())
                    * active_3d_fp32.unsqueeze(-1)
                )
                pred_acc_3d = (
                    torch.einsum("...ji,...kj->...ki", R_t, acc_cam.float())
                    * active_3d_fp32.unsqueeze(-1)
                )
            outputs["pred_vel_3d"] = pred_vel_3d
            outputs["pred_acc_3d"] = pred_acc_3d
            if self.predict_static_contact:
                outputs["static_contact_logits"] = (
                    self.static_contact_head(h_pva).float() * active_3d_fp32
                )

        return outputs


__all__ = [
    "MonocularEncoderRoPEBlock",
    "NUM_STATIC_CONTACT_JOINTS",
    "PLCSMonocularGVHMRModel",
    "STATIC_CONTACT_JOINT_INDICES",
    "build_monocular_sliding_padding_mask",
]
