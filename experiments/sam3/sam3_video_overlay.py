import argparse
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch

import sam3.model_builder as sam3_model_builder
from sam3.model_builder import build_sam3_video_predictor


def _ensure_sam3_bpe_vocab() -> Path:
    assets_dir = Path(sam3_model_builder.__file__).resolve().parent.parent / "assets"
    bpe_path = assets_dir / "bpe_simple_vocab_16e6.txt.gz"
    if bpe_path.exists():
        return bpe_path

    assets_dir.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
    urllib.request.urlretrieve(url, bpe_path)
    if not bpe_path.exists():
        raise FileNotFoundError(f"Failed to prepare BPE vocab at: {bpe_path}")
    return bpe_path


def run_sam3_video_inference(video_path: Path, prompt: str) -> None:
    """Run SAM3 video predictor and write an overlayed video.

    - 入力:  data/samples/clip.mp4
    - 出力:  data/samples/clip_sam3_overlay.mp4

    frame_index=0 にテキストプロンプトを与え、video tracking で全フレームへ伝播し、
    追跡された同一オブジェクトのマスクを全フレームにオーバーレイして保存する。
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sam3] Using device: {device}")

    # Sam3VideoPredictorMultiGPU は nn.Module ではなく、.to() は実装していない。
    # デバイス管理は内部で行われるため、そのままインスタンス化して利用する。
    _ensure_sam3_bpe_vocab()
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

    out_obj_ids = outputs.get("out_obj_ids")
    if out_obj_ids is None:
        raise KeyError("SAM3 outputs missing 'out_obj_ids'")

    # frame 0 におけるスコア最大の object id を「追跡対象」として固定する。
    best_idx = int(np.argmax(probs))
    target_obj_id = int(out_obj_ids[best_idx])
    best_mask = masks[best_idx]  # (H, W), np.ndarray
    print(
        f"[sam3] Selected target obj_id={target_obj_id} (index={best_idx}), "
        f"mask shape={best_mask.shape}, prob={float(probs[best_idx]):.3f}"
    )

    # OpenCV で入力動画を読み込み、各フレームに対応するマスクを適用して出力動画を書き出す。
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

    # オーバーレイ色（BGR）とアルファ
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
    alpha = 0.5

    def overlay_mask_on_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.shape != (height, width):
            resized = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bool_local = resized > 0.5
        else:
            mask_bool_local = mask.astype(bool)

        frame_float = frame.astype(np.float32)
        color_float = overlay_color.astype(np.float32)
        mask3 = mask_bool_local[:, :, None]
        blended = alpha * color_float[None, None, :] + (1.0 - alpha) * frame_float
        frame_float = np.where(mask3, blended, frame_float)
        return frame_float.astype(np.uint8)

    try:
        # --- 全フレームに伝播（tracking）して出力をストリームで受け取る ---
        if not hasattr(video_predictor, "handle_stream_request"):
            raise RuntimeError(
                "This sam3 video predictor does not support handle_stream_request; "
                "cannot run full-video propagation."
            )

        current_idx = 0
        for resp in video_predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ):
            resp_frame_idx = int(resp["frame_index"])
            resp_outputs = resp["outputs"]

            # resp_frame_idx まで動画を読み進めつつ書き出す
            while current_idx <= resp_frame_idx:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_idx == resp_frame_idx:
                    frame_masks = resp_outputs.get("out_binary_masks")
                    frame_obj_ids = resp_outputs.get("out_obj_ids")
                    if (
                        isinstance(frame_masks, np.ndarray)
                        and isinstance(frame_obj_ids, np.ndarray)
                        and frame_masks.ndim == 3
                        and frame_obj_ids.ndim == 1
                    ):
                        matches = np.where(frame_obj_ids == target_obj_id)[0]
                        if matches.size > 0:
                            frame = overlay_mask_on_frame(
                                frame, frame_masks[int(matches[0])]
                            )

                writer.write(frame)
                current_idx += 1

        # 念のため残りフレームを書き出し（通常は current_idx == frame_count で終わる想定）
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            current_idx += 1

        # NOTE: sam3==0.1.2 では end_session が未対応のため呼ばない。
        print(f"[sam3] Saved overlay video to: {output_path}")
    finally:
        cap.release()
        writer.release()


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

    video_path = args.video
    if not video_path.exists():
        fallback = Path("/root/repos/tennis-lab/data/samples/clip.mp4")
        if fallback.exists():
            video_path = fallback

    run_sam3_video_inference(video_path=video_path, prompt=args.prompt)


if __name__ == "__main__":
    main()
