import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor


def run_sam3_video_inference(video_path: Path, prompt: str) -> None:
    """Run SAM3 video predictor and write an overlayed video.

    - 入力:  data/samples/clip.mp4
    - 出力:  data/samples/clip_sam3_overlay.mp4

    現状は frame_index=0 のマスクを同じフレームにオーバーレイする実装。
    SAM3 の video API による全フレーム伝播は、仕様が分かり次第拡張する。
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sam3] Using device: {device}")

    # Sam3VideoPredictorMultiGPU は nn.Module ではなく、.to() は実装していない。
    # デバイス管理は内部で行われるため、そのままインスタンス化して利用する。
    video_predictor = build_sam3_video_predictor()

    # Start a session on the video
    print(f"[sam3] Starting session for video: {video_path}")
    response = video_predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(video_path),
        }
    )
    print(f"[sam3] start_session response keys: {list(response.keys())}")

    session_id = response.get("session_id")
    if session_id is None:
        raise RuntimeError(f"start_session did not return session_id: {response}")

    # Add a text prompt on the first frame.
    frame_index = 0
    print(f"[sam3] Adding text prompt '{prompt}' at frame_index={frame_index}")
    response = video_predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
    )

    print(f"[sam3] add_prompt response keys: {list(response.keys())}")
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError(f"Unexpected outputs type from SAM3: {type(outputs)}")

    # outputs の構造: dict で以下のキーを持つことを確認済み
    # - out_obj_ids: (N,)
    # - out_probs: (N,)
    # - out_boxes_xywh: (N, 4)
    # - out_binary_masks: (N, H, W)
    # - frame_stats: dict
    masks = outputs.get("out_binary_masks")
    probs = outputs.get("out_probs")
    if masks is None or probs is None:
        raise KeyError("SAM3 outputs missing 'out_binary_masks' or 'out_probs'")

    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks.shape}")

    # スコア最大のオブジェクトだけを使う（テニスプレーヤー想定）。
    best_idx = int(np.argmax(probs))
    best_mask = masks[best_idx]  # (H, W), np.ndarray
    print(
        f"[sam3] Selected object index {best_idx} for overlay, "
        f"mask shape={best_mask.shape}, prob={float(probs[best_idx]):.3f}"
    )

    # OpenCV で入力動画を読み込み、マスクを frame_index に適用して出力動画を書き出す。
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[sam3] Input video: {frame_count} frames, {width}x{height}, "
        f"fps={fps:.2f}"
    )

    output_path = video_path.parent / "clip_sam3_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    # マスクの解像度が動画と異なる場合はリサイズ。
    if best_mask.shape != (height, width):
        print(
            f"[sam3] Resizing mask from {best_mask.shape} to "
            f"({height}, {width}) for overlay."
        )
        resized_mask = cv2.resize(
            best_mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_bool = resized_mask > 0.5
    else:
        mask_bool = best_mask.astype(bool)

    # オーバーレイ色（BGR）とアルファ
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
    alpha = 0.5

    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx == frame_index:
            # マスク位置だけ overlay_color とブレンド
            frame_float = frame.astype(np.float32)
            color_float = overlay_color.astype(np.float32)

            # (H, W, 1) のブールマスクを作成し、チャンネル方向にブロードキャスト
            mask3 = mask_bool[:, :, None]
            blended = (
                alpha * color_float[None, None, :]
                + (1.0 - alpha) * frame_float
            )
            frame_float = np.where(mask3, blended, frame_float)
            frame = frame_float.astype(np.uint8)

        writer.write(frame)
        current_idx += 1

    cap.release()
    writer.release()

    print(f"[sam3] Saved overlay video to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/samples/clip.mp4"),
        help="Path to input video (MP4)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="tennis player",
        help="Text prompt for SAM3",
    )
    args = parser.parse_args()

    run_sam3_video_inference(video_path=args.video, prompt=args.prompt)


if __name__ == "__main__":
    main()
