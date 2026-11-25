"""HTD-Refine Stage B: Higher-order Temporal Dynamics 3D skeleton trajectory refiner.

Adapts the core principles of HTD-Refine (``arXiv:2605.26879``) to court-space
COCO-17 3D tennis player trajectories:

1. Velocity & Acceleration matching (``L_V + L_A``) against Stage A ``PVA-Heads``
   predictions, suppressing high-frequency 2D/bbox jitter while preserving sharp
   split-step and direction-change acceleration peaks.
2. Confidence-weighted 2D pinhole reprojection consistency (``L_reproj``) supporting
   both monocular (``V = 1``) and multi-camera (``V >= 2``) calibrated views
   without retraining.
3. Anatomical time-invariant bone-length constancy & L/R limb symmetry (``L_bone``).
4. Court-plane static contact zero-velocity & ground non-penetration (``L_court``).
5. Strict gap & padding boundary isolation: finite differences and segment bone
   statistics never couple across ``~valid_frame_mask`` gaps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.plcs.geometry.monocular_camera_bbox import (
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    validate_pinhole_camera_tensors,
)
from src.tasks.plcs.models.plcs_monocular_gvhmr_model import (
    STATIC_CONTACT_JOINT_INDICES,
)
from src.utils.geometry.court_pose import world_pose_to_canonical_pose
from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.schema.court_normalization import normalize_court_position
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES, NUM_HUMAN_KP

_SYMMETRIC_BONE_EDGE_PAIRS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((5, 11), (6, 12)),   # torso sides
    ((5, 7), (6, 8)),     # upper arms
    ((7, 9), (8, 10)),    # forearms
    ((11, 13), (12, 14)), # thighs
    ((13, 15), (14, 16)), # shanks
)


@dataclass(frozen=True, slots=True)
class HTDRefineCameraView:
    """Calibrated camera view + pixel 2D COCO-17 observations for Stage B refinement."""

    R_w2c: Tensor
    C_w: Tensor
    fx: float | Tensor
    fy: float | Tensor
    cx0: float | Tensor
    cy0: float | Tensor
    image_w: float
    image_h: float
    kp2d_px: Tensor
    kp_conf: Tensor
    frame_observed: Tensor

    def __post_init__(self) -> None:
        device = self.R_w2c.device
        dtype = self.R_w2c.dtype
        fx_t = torch.as_tensor(self.fx, device=device, dtype=dtype)
        fy_t = torch.as_tensor(self.fy, device=device, dtype=dtype)
        cx_t = torch.as_tensor(self.cx0, device=device, dtype=dtype)
        cy_t = torch.as_tensor(self.cy0, device=device, dtype=dtype)
        w_t = torch.as_tensor(self.image_w, device=device, dtype=dtype)
        h_t = torch.as_tensor(self.image_h, device=device, dtype=dtype)
        validate_pinhole_camera_tensors(
            self.R_w2c,
            self.C_w,
            fx=fx_t,
            fy=fy_t,
            cx0=cx_t,
            cy0=cy_t,
            image_w=w_t,
            image_h=h_t,
        )
        if self.kp2d_px.numel() > 0:
            if self.kp2d_px.ndim != 3 or self.kp2d_px.shape[1:] != (NUM_HUMAN_KP, 2):
                raise ValueError(
                    f"kp2d_px must have shape (T, 17, 2), got {tuple(self.kp2d_px.shape)}."
                )
            seq_len = self.kp2d_px.shape[0]
            if self.kp_conf.shape != (seq_len, NUM_HUMAN_KP):
                raise ValueError("kp_conf must have shape (T, 17).")
            if (
                self.frame_observed.shape != (seq_len,)
                or self.frame_observed.dtype != torch.bool
            ):
                raise ValueError("frame_observed must be bool of shape (T,).")
            if not torch.isfinite(self.kp2d_px).all() or not torch.isfinite(
                self.kp_conf
            ).all():
                raise ValueError("kp2d_px and kp_conf must contain only finite values.")


@dataclass(frozen=True, slots=True)
class HTDRefinerConfig:
    """Hyperparameters for ``PLCSHTDRefiner``."""

    num_steps: int = 40
    lr: float = 1.0
    vel_weight: float = 4.0
    acc_weight: float = 2.0
    reproj_weight: float = 4.0
    bone_var_weight: float = 8.0
    bone_sym_weight: float = 3.0
    contact_weight: float = 4.0
    ground_penetration_weight: float = 12.0
    jerk_weight: float = 0.05
    init_anchor_weight: float = 0.05
    vel_huber_delta: float = 2.0
    acc_huber_delta: float = 20.0
    reproj_huber_delta: float = 0.25
    conf_threshold: float = 0.2
    min_joint_z_m: float = 0.0
    bone_var_per_segment: bool = True

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative.")
        if not (math.isfinite(self.lr) and self.lr > 0.0):
            raise ValueError("lr must be positive and finite.")
        for name in (
            "vel_weight",
            "acc_weight",
            "reproj_weight",
            "bone_var_weight",
            "bone_sym_weight",
            "contact_weight",
            "ground_penetration_weight",
            "jerk_weight",
            "init_anchor_weight",
        ):
            val = getattr(self, name)
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class HTDRefineResult:
    """Refined court-space 3D trajectory and derived PLCS placement tensors."""

    world_kp_3d: Tensor
    position: Tensor
    rotation: Tensor
    canonical_pose: Tensor
    valid_frame_mask: Tensor
    objective_history: tuple[float, ...]


def _contiguous_true_segments(mask: Tensor) -> list[tuple[int, int]]:
    """Return ``[start, end)`` index ranges of contiguous ``True`` runs in 1D bool mask."""
    segments: list[tuple[int, int]] = []
    n = int(mask.numel())
    in_seg = False
    start = 0
    for idx, val in enumerate(mask.tolist()):
        if val and not in_seg:
            start = idx
            in_seg = True
        elif not val and in_seg:
            segments.append((start, idx))
            in_seg = False
    if in_seg:
        segments.append((start, n))
    return segments


class PLCSHTDRefiner:
    """Stage B sequence-level 3D COCO-17 trajectory optimizer."""

    def __init__(self, config: HTDRefinerConfig | None = None) -> None:
        self.config = HTDRefinerConfig() if config is None else config

    def refine(
        self,
        *,
        world_kp_init: Tensor,
        pred_vel_3d: Tensor | None,
        pred_acc_3d: Tensor | None,
        static_contact_logits: Tensor | None,
        views: Sequence[HTDRefineCameraView],
        fps: float,
        valid_frame_mask: Tensor | None = None,
        vel_valid_mask: Tensor | None = None,
        acc_valid_mask: Tensor | None = None,
    ) -> HTDRefineResult:
        """Optimize ``world_kp_3d`` ``(T, 17, 3)`` [m] without crossing invalid/padded gaps."""
        if world_kp_init.ndim != 3 or world_kp_init.shape[1:] != (NUM_HUMAN_KP, 3):
            raise ValueError(
                f"world_kp_init must have shape (T, 17, 3), got {tuple(world_kp_init.shape)}."
            )
        if not (math.isfinite(fps) and fps > 0.0):
            raise ValueError(f"fps must be positive and finite, got {fps}.")

        seq_len = int(world_kp_init.shape[0])
        device = world_kp_init.device
        dtype = world_kp_init.dtype

        if valid_frame_mask is None:
            valid_mask = torch.ones((seq_len,), dtype=torch.bool, device=device)
        else:
            if (
                valid_frame_mask.shape != (seq_len,)
                or valid_frame_mask.dtype != torch.bool
            ):
                raise ValueError("valid_frame_mask must be bool of shape (T,).")
            valid_mask = valid_frame_mask.to(device=device)

        vel_mask_t: Tensor | None = None
        if vel_valid_mask is not None:
            if vel_valid_mask.shape != (seq_len,) or vel_valid_mask.dtype != torch.bool:
                raise ValueError("vel_valid_mask must be bool of shape (T,).")
            vel_mask_t = vel_valid_mask.to(device=device)

        acc_mask_t: Tensor | None = None
        if acc_valid_mask is not None:
            if acc_valid_mask.shape != (seq_len,) or acc_valid_mask.dtype != torch.bool:
                raise ValueError("acc_valid_mask must be bool of shape (T,).")
            acc_mask_t = acc_valid_mask.to(device=device)

        if not torch.isfinite(world_kp_init[valid_mask]).all():
            raise ValueError("world_kp_init contains non-finite coordinates on valid frames.")

        cfg = self.config
        if cfg.num_steps == 0 or not bool(valid_mask.any()):
            return self._pack_result(world_kp_init.detach().clone(), valid_mask, ())

        # Detach fixed optimization targets cleanly and zero invalid frames so
        # NaNs/outliers in invalid frames can never enter the computation graph.
        base_kp = torch.where(
            valid_mask[:, None, None],
            world_kp_init.detach(),
            torch.zeros_like(world_kp_init),
        )
        vel_target = (
            pred_vel_3d.detach().to(device=device, dtype=dtype)
            if pred_vel_3d is not None
            else None
        )
        acc_target = (
            pred_acc_3d.detach().to(device=device, dtype=dtype)
            if pred_acc_3d is not None
            else None
        )
        contact_probs = (
            torch.sigmoid(static_contact_logits.detach().to(device=device, dtype=dtype))
            if static_contact_logits is not None
            else None
        )

        segments = _contiguous_true_segments(valid_mask)
        edge_a = [e[0] for e in COCO17_BONE_LENGTH_EDGES]
        edge_b = [e[1] for e in COCO17_BONE_LENGTH_EDGES]
        sym_l0 = [p[0][0] for p in _SYMMETRIC_BONE_EDGE_PAIRS]
        sym_l1 = [p[0][1] for p in _SYMMETRIC_BONE_EDGE_PAIRS]
        sym_r0 = [p[1][0] for p in _SYMMETRIC_BONE_EDGE_PAIRS]
        sym_r1 = [p[1][1] for p in _SYMMETRIC_BONE_EDGE_PAIRS]

        history: list[float] = []
        dt = 1.0 / float(fps)
        vel_delta_m = max(cfg.vel_huber_delta * dt, 1e-3)
        acc_delta_m = max(cfg.acc_huber_delta * dt * dt, 1e-3)

        def _slice_cam_tensor(tensor: Tensor, start: int, end: int, *, time_ndim: int) -> Tensor:
            return tensor[start:end] if tensor.ndim == time_ndim else tensor

        final_kp = world_kp_init.detach().clone()

        # Optimize each contiguous valid segment independently so missing/padding
        # gaps are 100% isolated (including optimizer state and L-BFGS curvature).
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            seg_base = base_kp[seg_start:seg_end]
            seg_vel = vel_target[seg_start:seg_end] if vel_target is not None else None
            seg_acc = acc_target[seg_start:seg_end] if acc_target is not None else None
            seg_vel_mask = (
                vel_mask_t[seg_start:seg_end] if vel_mask_t is not None else None
            )
            seg_acc_mask = (
                acc_mask_t[seg_start:seg_end] if acc_mask_t is not None else None
            )
            seg_contact = (
                contact_probs[seg_start:seg_end] if contact_probs is not None else None
            )

            prepared_views: list[
                tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            ] = []
            if cfg.reproj_weight > 0.0 and views:
                for view in views:
                    if view.kp2d_px.numel() == 0:
                        continue
                    obs = view.frame_observed[seg_start:seg_end].to(device=device)
                    conf = view.kp_conf[seg_start:seg_end].to(device=device, dtype=dtype)
                    joint_mask = obs[:, None] & (conf > cfg.conf_threshold)
                    if not bool(joint_mask.any()):
                        continue
                    fx_t = _slice_cam_tensor(
                        torch.as_tensor(view.fx, device=device, dtype=dtype),
                        seg_start,
                        seg_end,
                        time_ndim=1,
                    )
                    fy_t = _slice_cam_tensor(
                        torch.as_tensor(view.fy, device=device, dtype=dtype),
                        seg_start,
                        seg_end,
                        time_ndim=1,
                    )
                    cx_t = _slice_cam_tensor(
                        torch.as_tensor(view.cx0, device=device, dtype=dtype),
                        seg_start,
                        seg_end,
                        time_ndim=1,
                    )
                    cy_t = _slice_cam_tensor(
                        torch.as_tensor(view.cy0, device=device, dtype=dtype),
                        seg_start,
                        seg_end,
                        time_ndim=1,
                    )
                    R_seg = _slice_cam_tensor(
                        view.R_w2c.to(device=device, dtype=dtype),
                        seg_start,
                        seg_end,
                        time_ndim=3,
                    )
                    C_seg = _slice_cam_tensor(
                        view.C_w.to(device=device, dtype=dtype),
                        seg_start,
                        seg_end,
                        time_ndim=2,
                    )
                    R_in = R_seg.unsqueeze(0)
                    C_in = C_seg.unsqueeze(0)
                    fx_in = fx_t.unsqueeze(0)
                    fy_in = fy_t.unsqueeze(0)
                    cx_in = cx_t.unsqueeze(0)
                    cy_in = cy_t.unsqueeze(0)
                    kp2d_t = view.kp2d_px[seg_start:seg_end].to(device=device, dtype=dtype)
                    prepared_views.append(
                        (R_in, C_in, fx_in, fy_in, cx_in, cy_in, kp2d_t, conf, joint_mask)
                    )

            with torch.enable_grad():
                delta = torch.nn.Parameter(torch.zeros_like(seg_base))
                optimizer = torch.optim.LBFGS(
                    [delta],
                    lr=1.0,
                    max_iter=cfg.num_steps,
                    history_size=20,
                    line_search_fn="strong_wolfe",
                )

                def _closure(
                    *,
                    _seg_base: Tensor = seg_base,
                    _delta: Tensor = delta,
                    _seg_len: int = seg_len,
                    _seg_vel: Tensor | None = seg_vel,
                    _seg_acc: Tensor | None = seg_acc,
                    _seg_vel_mask: Tensor | None = seg_vel_mask,
                    _seg_acc_mask: Tensor | None = seg_acc_mask,
                    _seg_contact: Tensor | None = seg_contact,
                    _prepared_views: list[
                        tuple[
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                        ]
                    ] = prepared_views,
                    _optimizer: torch.optim.LBFGS = optimizer,
                ) -> Tensor:
                    _optimizer.zero_grad()
                    kp = _seg_base + _delta
                    total = kp.new_zeros(())

                    # 1. Stage A 1st-order forward velocity alignment in metres per step.
                    # Stage A supervises pred_vel[t] against (J[t+1] - J[t]) * fps, so
                    # pred_vel[:-1] * dt aligns identically with step_disp = kp[1:] - kp[:-1]
                    # and never reads the unsupervised segment-endpoint pred_vel[-1].
                    if cfg.vel_weight > 0.0 and _seg_vel is not None and _seg_len >= 2:
                        step_disp = kp[1:] - kp[:-1]
                        target_disp = _seg_vel[:-1] * dt
                        if _seg_vel_mask is None:
                            loss_v = F.huber_loss(
                                step_disp,
                                target_disp,
                                delta=vel_delta_m,
                            )
                            total = total + cfg.vel_weight * loss_v
                        else:
                            v_step_mask = _seg_vel_mask[:-1]
                            if bool(v_step_mask.any()):
                                loss_v = F.huber_loss(
                                    step_disp[v_step_mask],
                                    target_disp[v_step_mask],
                                    delta=vel_delta_m,
                                )
                                total = total + cfg.vel_weight * loss_v

                    # 2. Stage A 2nd-order acceleration alignment in metres per step^2
                    if cfg.acc_weight > 0.0 and _seg_acc is not None and _seg_len >= 3:
                        second_diff = kp[2:] - 2.0 * kp[1:-1] + kp[:-2]
                        target_dd = _seg_acc[1:-1] * (dt * dt)
                        if _seg_acc_mask is None:
                            loss_a = F.huber_loss(
                                second_diff,
                                target_dd,
                                delta=acc_delta_m,
                            )
                            total = total + cfg.acc_weight * loss_a
                        else:
                            a_step_mask = _seg_acc_mask[1:-1]
                            if bool(a_step_mask.any()):
                                loss_a = F.huber_loss(
                                    second_diff[a_step_mask],
                                    target_dd[a_step_mask],
                                    delta=acc_delta_m,
                                )
                                total = total + cfg.acc_weight * loss_a

                    # 3. Confidence-weighted metric 2D pinhole reprojection across 1..V views
                    # Uses per-axis focal lengths (fx, fy) for in-front joints and a smooth
                    # positive-depth cheirality barrier for behind-camera joints so gradients
                    # always push behind-camera points forward in front of the camera.
                    if cfg.reproj_weight > 0.0 and _prepared_views:
                        reproj_sum = kp.new_zeros(())
                        min_depth_m = 0.1
                        for (
                            R_in,
                            C_in,
                            fx_in,
                            fy_in,
                            cx_in,
                            cy_in,
                            kp2d_t,
                            conf,
                            joint_mask,
                        ) in _prepared_views:
                            R_v = R_in[0]
                            C_v = C_in[0]
                            if R_v.ndim == 2:
                                kp_cam = (kp - C_v.view(1, 1, 3)) @ R_v.T
                            else:
                                kp_cam = torch.matmul(
                                    kp - C_v.unsqueeze(1), R_v.transpose(-1, -2)
                                )
                            fx_v = fx_in[0]
                            fy_v = fy_in[0]
                            cx_v = cx_in[0]
                            cy_v = cy_in[0]
                            fx_col = fx_v.view(1, 1) if fx_v.ndim == 0 else fx_v.view(-1, 1)
                            fy_col = fy_v.view(1, 1) if fy_v.ndim == 0 else fy_v.view(-1, 1)
                            cx_col = cx_v.view(1, 1) if cx_v.ndim == 0 else cx_v.view(-1, 1)
                            cy_col = cy_v.view(1, 1) if cy_v.ndim == 0 else cy_v.view(-1, 1)

                            z_cam = kp_cam[..., 2]
                            in_front = z_cam > min_depth_m
                            z_safe = torch.where(in_front, z_cam, torch.ones_like(z_cam))
                            u_pred = fx_col * (kp_cam[..., 0] / z_safe) + cx_col
                            v_pred = fy_col * (kp_cam[..., 1] / z_safe) + cy_col
                            depth_det = z_safe.detach()
                            metric_err_x = (u_pred - kp2d_t[..., 0]) * (depth_det / fx_col)
                            metric_err_y = (v_pred - kp2d_t[..., 1]) * (depth_det / fy_col)
                            metric_err = torch.stack([metric_err_x, metric_err_y], dim=-1)
                            in_front_err = F.huber_loss(
                                metric_err,
                                torch.zeros_like(metric_err),
                                reduction="none",
                                delta=cfg.reproj_huber_delta,
                            ).mean(dim=-1)

                            # Differentiable positive-depth cheirality penalty for z_cam <= min_depth_m
                            depth_margin = F.relu((min_depth_m + 0.5) - z_cam)
                            behind_err = F.huber_loss(
                                depth_margin,
                                torch.zeros_like(depth_margin),
                                reduction="none",
                                delta=cfg.reproj_huber_delta,
                            ) + 0.5 * (depth_margin**2)

                            err = torch.where(in_front, in_front_err, behind_err)
                            weighted = err * conf
                            reproj_sum = reproj_sum + weighted[joint_mask].mean()
                        total = total + cfg.reproj_weight * (
                            reproj_sum / float(len(_prepared_views))
                        )

                    # 4. Segment-isolated bone-length temporal constancy & L/R symmetry
                    if cfg.bone_var_weight > 0.0 and _seg_len >= 2:
                        bones = torch.linalg.vector_norm(
                            kp[:, edge_a, :] - kp[:, edge_b, :], dim=-1
                        )
                        total = (
                            total
                            + cfg.bone_var_weight
                            * bones.var(dim=0, unbiased=False).mean()
                        )

                    if cfg.bone_sym_weight > 0.0:
                        len_l = torch.linalg.vector_norm(
                            kp[:, sym_l0, :] - kp[:, sym_l1, :],
                            dim=-1,
                        )
                        len_r = torch.linalg.vector_norm(
                            kp[:, sym_r0, :] - kp[:, sym_r1, :],
                            dim=-1,
                        )
                        total = total + cfg.bone_sym_weight * ((len_l - len_r) ** 2).mean()

                    # 5. Court-plane ground non-penetration & static foot/knee contact
                    if cfg.ground_penetration_weight > 0.0:
                        penetration = F.relu(cfg.min_joint_z_m - kp[..., 2])
                        total = (
                            total
                            + cfg.ground_penetration_weight * (penetration**2).mean()
                        )

                    if (
                        cfg.contact_weight > 0.0
                        and _seg_contact is not None
                        and _seg_len >= 2
                    ):
                        c_joints = kp[:, list(STATIC_CONTACT_JOINT_INDICES), :]
                        c_step = c_joints[1:] - c_joints[:-1]
                        c_prob_pair = 0.5 * (_seg_contact[1:] + _seg_contact[:-1])
                        foot_slide = (c_step**2).sum(dim=-1) * c_prob_pair
                        total = total + cfg.contact_weight * foot_slide.mean()

                    # 6. Light 3rd-order jerk prior + initial prediction anchor
                    if cfg.jerk_weight > 0.0 and _seg_len >= 4:
                        jerk = kp[3:] - 3.0 * kp[2:-1] + 3.0 * kp[1:-2] - kp[:-3]
                        total = total + cfg.jerk_weight * (jerk**2).mean()

                    if cfg.init_anchor_weight > 0.0:
                        total = total + cfg.init_anchor_weight * (_delta**2).mean()

                    total.backward()
                    history.append(float(total.detach().cpu().item()))
                    return total

                optimizer.step(_closure)

            final_kp[seg_start:seg_end] = (seg_base + delta.detach()).to(
                dtype=final_kp.dtype
            )

        refined_valid = final_kp[valid_mask]
        for check_fn in (
            compute_bone_lengths,
            compute_joint_angles,
            compute_torsion_angles,
            compute_torso_twist,
        ):
            check_fn(refined_valid).require_valid("PLCSHTDRefiner output")

        return self._pack_result(final_kp, valid_mask, tuple(history))

    @staticmethod
    def _pack_result(
        world_kp_3d: Tensor,
        valid_frame_mask: Tensor,
        history: tuple[float, ...],
    ) -> HTDRefineResult:
        valid = valid_frame_mask
        position = torch.zeros(
            (world_kp_3d.shape[0], 3), device=world_kp_3d.device, dtype=world_kp_3d.dtype
        )
        rotation = torch.zeros(
            (world_kp_3d.shape[0], 2), device=world_kp_3d.device, dtype=world_kp_3d.dtype
        )
        rotation[:, 0] = 1.0
        canonical_pose = torch.zeros_like(world_kp_3d)

        if bool(valid.any()):
            kp_v = world_kp_3d[valid]
            pelvis_v = 0.5 * (
                kp_v[..., LEFT_HIP_IDX, :] + kp_v[..., RIGHT_HIP_IDX, :]
            )
            pos_v = normalize_court_position(pelvis_v)
            trans_v = kp_v[..., LEFT_HIP_IDX, :2] - kp_v[..., RIGHT_HIP_IDX, :2]
            rot_v = F.normalize(trans_v, dim=-1, eps=1e-6)
            canon_v = world_pose_to_canonical_pose(kp_v, pos_v, rot_v)
            position[valid] = pos_v
            rotation[valid] = rot_v
            canonical_pose[valid] = canon_v

        return HTDRefineResult(
            world_kp_3d=world_kp_3d,
            position=position,
            rotation=rotation,
            canonical_pose=canonical_pose,
            valid_frame_mask=valid_frame_mask,
            objective_history=history,
        )


__all__ = [
    "HTDRefineCameraView",
    "HTDRefineResult",
    "HTDRefinerConfig",
    "PLCSHTDRefiner",
]
