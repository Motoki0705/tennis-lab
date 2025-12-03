#!/usr/bin/env python
"""AMASS (SMPL-H) の npz から 3D ジョイントを計算 & 可視化するデモ.

- 入力: AMASS の 1 シーケンス npz / 既存の *_joints3d.npz
- 出力: joints (T, J, 3) [m] を計算して保存し、希望に応じてレンダリング
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import smplx
import torch


def load_amass_sequence(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    poses = data["poses"].astype(np.float32)  # (T, 156)
    trans = data["trans"].astype(np.float32)  # (T, 3)
    betas = data["betas"].astype(np.float32)  # (16,)
    gender = str(data["gender"].tolist())  # b'female' → "female" など
    fps = float(data.get("mocap_framerate", np.array([60.0], dtype=np.float32)).item())

    return poses, trans, betas, gender, fps


def split_smplh_poses(poses: np.ndarray):
    """poses: (T, 156)  = 52 * 3 [axis-angle]

    AMASS/SMPL-H の構造:
      1   *3: global orientation
      21  *3: body
      15  *3: right hand
      15  *3: left hand
    """
    T = poses.shape[0]

    aa = poses.reshape(T, 52, 3)  # (T, 52, 3)

    global_orient = aa[:, 0]  # (T, 3)
    body_pose = aa[:, 1:22]  # (T, 21, 3)

    right_hand_pose = aa[:, 22:37]  # (T, 15, 3)
    left_hand_pose = aa[:, 37:52]  # (T, 15, 3)

    return global_orient, body_pose, left_hand_pose, right_hand_pose


def build_smplh_model(model_root: Path, gender: str, num_betas: int):
    """smplx.create を使って SMPL-H モデルを構築"""
    model = smplx.create(
        model_path=str(model_root),
        model_type="smplh",  # ← ここが重要
        gender=gender.lower(),  # "male" / "female" / "neutral"
        num_betas=num_betas,
        use_pca=False,  # 手指を 15*3 の axis-angle で渡すため PCA を無効化
        ext="pkl",  # SMPLH_FEMALE.pkl などを読む
    )
    # バッチサイズは forward 時に勝手に合わせてくれるので、ここでは特に指定不要。
    return model


def render_joints_matplotlib(
    joints_np: np.ndarray,
    parents: list[int],
    fps: float,
    frame_stride: int = 1,
):
    """Matplotlib で簡易 3D 可視化"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib が必要です。pip install matplotlib を実行してください。"
        ) from exc

    if joints_np.ndim != 3:
        raise ValueError("joints_np は (T, J, 3) である必要があります")

    stride = max(1, int(frame_stride))
    frames = joints_np[::stride]
    if frames.shape[0] == 0:
        raise ValueError("フレームが存在しません (stride が大きすぎる可能性)")

    parents_arr = np.asarray(parents)
    if parents_arr.shape[0] < frames.shape[1]:
        pad = np.full(
            frames.shape[1] - parents_arr.shape[0], -1, dtype=parents_arr.dtype
        )
        parents_arr = np.concatenate([parents_arr, pad], axis=0)
    xyz = frames.reshape(-1, 3)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = (xyz_max + xyz_min) / 2.0
    extent = (xyz_max - xyz_min).max()
    extent = float(extent if extent > 0 else 1.0)
    padding = extent * 0.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("SMPL-H joints")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    scat = ax.scatter([], [], [], c="tab:red", s=15)
    bones = []
    for idx in range(1, frames.shape[1]):
        parent = int(parents_arr[idx])
        if parent < 0 or parent >= frames.shape[1]:
            continue
        (line,) = ax.plot([], [], [], c="tab:blue", linewidth=1.5)
        bones.append((line, idx, parent))

    def init():
        ax.set_xlim(center[0] - extent - padding, center[0] + extent + padding)
        ax.set_ylim(center[1] - extent - padding, center[1] + extent + padding)
        ax.set_zlim(center[2] - extent - padding, center[2] + extent + padding)
        pts0 = frames[0]
        scat._offsets3d = (pts0[:, 0], pts0[:, 1], pts0[:, 2])
        for line, child, parent in bones:
            xs = [pts0[parent, 0], pts0[child, 0]]
            ys = [pts0[parent, 1], pts0[child, 1]]
            zs = [pts0[parent, 2], pts0[child, 2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        return [scat] + [line for line, _, _ in bones]

    def update(frame_idx):
        pts = frames[frame_idx]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        artists = [scat]
        for line, child, parent in bones:
            xs = [pts[parent, 0], pts[child, 0]]
            ys = [pts[parent, 1], pts[child, 1]]
            zs = [pts[parent, 2], pts[child, 2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            artists.append(line)
        return artists

    interval_ms = max(1.0, 1000.0 * stride / max(1e-6, fps))
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames.shape[0],
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    plt.show()
    return anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amass_npz",
        type=Path,
        default="data/ACCAD/Female1General_c3d/A1 - Stand_poses.npz",
        help="AMASS の *_poses.npz",
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default="data/smplx",
        help="smplx のモデルルート (smplh/SMPLH_*.pkl が入っているディレクトリ)",
    )
    parser.add_argument(
        "--joints_npz",
        type=Path,
        help="計算済み *_joints3d.npz を直接読み込む場合に指定",
    )
    parser.add_argument(
        "--save_joints_npz",
        action="store_true",
        help="計算済み *_joints3d.npz を直接読み込む場合に指定",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="matplotlib を用いてジョイントを可視化",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="可視化時に間引くフレーム間隔",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joints_np: np.ndarray | None = None
    parents: list[int] | None = None
    fps: float = 60.0

    def _ensure_parent_length(
        parents_list: list[int] | None, joint_count: int
    ) -> list[int] | None:
        if parents_list is None:
            return None
        if len(parents_list) < joint_count:
            pad_len = joint_count - len(parents_list)
            print(
                f"WARNING: parents has {len(parents_list)} entries but joints need {joint_count}; padding with -1."
            )
            parents_list = parents_list + ([-1] * pad_len)
        return parents_list

    if args.joints_npz is not None:
        # 既存 joints を読み込み
        data = np.load(args.joints_npz, allow_pickle=True)
        joints_np = data["joints"].astype(np.float32)
        fps = float(data.get("fps", np.array([fps], dtype=np.float32)).item())
        if "parents" in data:
            parents = data["parents"].astype(np.int32).tolist()
        if "joint_names" in data:
            joint_names = data["joint_names"].tolist()
        else:
            joint_names = None
        print(f"loaded joints from {args.joints_npz} -> {joints_np.shape}")
        parents = _ensure_parent_length(parents, joints_np.shape[1])
    else:
        # 1) AMASS を読み込み
        poses_np, trans_np, betas_np, gender, fps = load_amass_sequence(args.amass_npz)
        T = poses_np.shape[0]
        print(f"frames: {T}, gender: {gender}")
        print(
            f"poses shape: {poses_np.shape}, trans shape: {trans_np.shape}, betas: {betas_np.shape}"
        )

        # 2) poses を global/body/hand に分解
        global_orient_np, body_pose_np, left_hand_np, right_hand_np = split_smplh_poses(
            poses_np
        )

        # 3) torch Tensor に変換
        global_orient = torch.from_numpy(global_orient_np).to(device)  # (T, 3)
        body_pose = torch.from_numpy(body_pose_np).to(device).reshape(T, -1)  # (T, 63)
        left_hand_pose = (
            torch.from_numpy(left_hand_np).to(device).reshape(T, -1)
        )  # (T, 45)
        right_hand_pose = (
            torch.from_numpy(right_hand_np).to(device).reshape(T, -1)
        )  # (T, 45)
        transl = torch.from_numpy(trans_np).to(device)  # (T, 3)

        # モデルに存在する shape 係数数 (SMPL-H の pkl は 10 が多い) に合わせて切り詰め
        num_betas_model = min(betas_np.shape[0], 10)
        if betas_np.shape[0] > num_betas_model:
            print(
                f"WARNING: betas has {betas_np.shape[0]} dims but model supports only {num_betas_model}; truncating."
            )

        # betas は 1人分なのでフレーム方向に broadcast
        betas = (
            torch.from_numpy(betas_np[None, :num_betas_model]).to(device).repeat(T, 1)
        )  # (T, num_betas_model)

        # 4) SMPL-H モデルを構築
        model = build_smplh_model(
            args.model_root, gender, num_betas=num_betas_model
        ).to(device)
        model.eval()

        # 5) forward してジョイントを取得
        with torch.no_grad():
            output = model(
                betas=betas,
                global_orient=global_orient,  # (T,3) axis-angle
                body_pose=body_pose,  # (T,21,3) axis-angle
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,  # (T,3)
                return_verts=False,
            )

        joints = output.joints  # (T, J, 3), 単位: m
        print(f"joints shape: {joints.shape}")

        # 例として、最初のフレームの腰(0番ジョイント)を表示
        j0 = joints[0, 0].cpu().numpy()
        print(f"frame0 pelvis: {j0}")

        joints_np = joints.cpu().numpy()
        parents = model.parents.tolist()
        parents = _ensure_parent_length(parents, joints_np.shape[1])
        joint_names = getattr(model, "joint_names", None)

        # 必要なら npz で保存 (メタ情報も含む)
        if args.save_joints_npz:
            out_path = args.amass_npz.with_name(args.amass_npz.stem + "_joints3d.npz")
            payload = {
                "joints": joints_np,
                "fps": np.array([fps], dtype=np.float32),
                "parents": np.array(
                    parents if parents is not None else [], dtype=np.int32
                ),
            }
            if joint_names is not None:
                payload["joint_names"] = np.array(joint_names)
            np.savez_compressed(out_path, **payload)
            print(f"saved joints to: {out_path}")

    # レンダリングオプション
    if args.render:
        if joints_np is None:
            raise RuntimeError("レンダリング対象 joints がありません")

        if parents is None:
            print("parents 情報が無いため neutral モデルから取得します")
            tmp_model = build_smplh_model(args.model_root, "neutral", num_betas=10)
            parents = tmp_model.parents.tolist()

        render_joints_matplotlib(
            joints_np=joints_np,
            parents=parents,
            fps=fps,
            frame_stride=args.frame_stride,
        )


if __name__ == "__main__":
    main()
