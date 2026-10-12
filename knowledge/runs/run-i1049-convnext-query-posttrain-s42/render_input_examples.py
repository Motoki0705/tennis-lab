"""Reproduce the issue #1049 input illustration from the frozen validation window.

Run with PYTHONPATH=. and the repository .venv. This is an evidence renderer,
not a training entry point; all image transforms use the production implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.data.windows.dataset import (
    FrozenBallWindowDataset,
    collate_window_samples,
)
from src.tasks.ball_detection.preprocessing.mdd import RGBToMDD

PROFILES = ("clean", "camera", "occlusion", "combined")
TITLES = ("元入力", "カメラ移動", "人工遮蔽", "カメラ移動＋人工遮蔽")
BG, FG, MUTED = "#101b2b", "#f3f7fc", "#bac9dc"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size, index=0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mdd_image(mdd: torch.Tensor) -> np.ndarray:
    """Fixed [0,1] for every profile/frame: red=darken, green=brighten."""
    values = mdd.detach().cpu().numpy()
    colors = np.stack((values[1], values[0], np.zeros_like(values[0])), axis=-1)
    return (colors * 255).round().astype(np.uint8)


def overview(cases: dict[str, Any], batch: dict[str, Any], index: int, kind: str) -> Image.Image:
    width, height, margin, gap = 560, 315, 24, 16
    canvas = Image.new("RGB", (1184, 928), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), "入力の変化  /  " + ("RGB" if kind == "rgb" else "MDD（モデルが読む2チャネル）"), font=font(30, bold=True), fill=FG)
    draw.text((margin, 62), f"TrackNet game9 / Clip2  |  32枚・frame step 4  |  入力時刻 {batch['timestamps'][0, index]:.3f} 秒", font=font(20), fill=MUTED)
    for n, profile in enumerate(PROFILES):
        case = cases[profile]
        x, y = margin + (n % 2) * (width + gap), 110 + (n // 2) * 382
        draw.text((x, y), TITLES[n], font=font(23, bold=True), fill=FG)
        pixels = case[kind][index]
        frame = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(frame, (x, y + 38))
        hidden = case["hidden"][index]
        text = "GT中心：人工遮蔽中・座標教師は保持" if hidden else "GT中心：人工遮蔽なし"
        draw.text((x, y + 354), text, font=font(17), fill="#ffd182" if hidden else MUTED)
    if kind == "mdd":
        draw.text((margin, 879), "緑 = 明るくなる差分 (M+)   赤 = 暗くなる差分 (M−)   全条件で同じ [0, 1] スケール", font=font(19), fill=FG)
    else:
        draw.text((margin, 879), "画像への注釈描画なし  |  全32枚を4fpsでゆっくり表示  |  実入力は30→7.5fps", font=font(19), fill=FG)
    return canvas


def save_gif(frames: list[Image.Image], path: Path) -> None:
    # One palette across time avoids frame-dependent color flicker.
    sheet = Image.new("RGB", (592 * 4, 464 * 4), BG)
    for i, frame in enumerate(frames[::2]):
        sheet.paste(frame.resize((592, 464)), ((i % 4) * 592, (i // 4) * 464))
    palette = sheet.quantize(colors=240, method=Image.Quantize.MEDIANCUT)
    frames_p = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=250, loop=0, optimize=True, disposal=1)


def occlusion_detail(cases: dict[str, Any], batch: dict[str, Any]) -> tuple[Image.Image, list[int]]:
    changed = np.flatnonzero(np.any(cases["occlusion"]["rgb"] != cases["clean"]["rgb"], axis=(1, 2, 3)))
    hidden = np.flatnonzero(cases["occlusion"]["hidden"])
    if not len(hidden):
        raise ValueError("This illustration must include a known GT center inside the sampled occluder")
    selected = list(dict.fromkeys([max(0, int(changed[0]) - 1), int(hidden[0]), int(changed[-1]), min(31, int(changed[-1]) + 1)]))
    if len(selected) != 4:
        raise ValueError("Need distinct before / hidden / final masked / after samples")
    canvas = Image.new("RGB", (1480, 1125), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "人工遮蔽の前・最中・解除後を拡大", font=font(31, bold=True), fill=FG)
    draw.text((24, 65), "全列で同じ画像位置192×144 pixelを拡大。水色の輪＝GT（範囲内のみ／表示専用・入力には含めない）。", font=font(20), fill=MUTED)
    titles = ("遮蔽前", "GT中心が遮蔽された時", "矩形の最終フレーム", "矩形を解除した後")
    row_names = ("元RGB", "遮蔽後RGB", "遮蔽後MDD")
    native_h, native_w = cases["clean"]["rgb"].shape[1:3]
    anchor_x, anchor_y = batch["uv"][0, int(hidden[0])].numpy() * np.array((native_w - 1, native_h - 1))
    left = int(np.clip(round(anchor_x) - 96, 0, native_w - 192))
    top = int(np.clip(round(anchor_y) - 72, 0, native_h - 144))
    for col, t in enumerate(selected):
        x = 24 + col * 365
        draw.text((x, 108), titles[col], font=font(20, bold=True), fill=FG)
        draw.text((x, 140), f"frame {batch['frame_indices'][0, t]}  /  {batch['timestamps'][0, t]:.3f}s", font=font(19), fill=MUTED)
        uv = batch["uv"][0, t].numpy()
        px, py = uv * np.array((native_w - 1, native_h - 1))
        for row, (profile, kind) in enumerate((("clean", "rgb"), ("occlusion", "rgb"), ("occlusion", "mdd"))):
            y = 181 + row * 284
            draw.text((x, y), row_names[row], font=font(18), fill=MUTED)
            pixels = cases[profile][kind][t, top:top + 144, left:left + 192]
            tile = Image.fromarray(pixels).resize((336, 252), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y + 30))
            if left <= px < left + 192 and top <= py < top + 144:
                cx, cy = x + (px - left) * 336 / 192, y + 30 + (py - top) * 252 / 144
                draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline="#46dff2", width=2)
    draw.text((24, 1060), "先頭MDDは0。遮蔽物はボールを追いません。解除直後は矩形自体の消失も差分として現れます。", font=font(20), fill=MUTED)
    return canvas, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((args.run / "config.json").read_text())
    recipe = config["recipe"]
    for current, old in (
        ("src/tasks/ball_detection/data/augmentation/video.py", "src/tasks/ball_detection/training/posttraining/augmentation.py"),
        ("src/tasks/ball_detection/preprocessing/mdd.py", "src/tasks/ball_detection/preprocessing/mdd.py"),
    ):
        if digest(Path(current)) != config["code"]["source_sha256"][old]:
            raise ValueError(f"Illustration source differs from the recorded run: {current}")
    manifest = args.run / "data_manifest.json"
    if digest(manifest) != recipe["manifest_sha256"]:
        raise ValueError("Run's saved manifest differs from the training recipe")
    data = FrozenBallWindowDataset(manifest, split="val", requires_pose=False, jpeg_decoder="opencv")
    index = next(i for i, (ri, window) in enumerate(data.windows)
                 if data.records[ri]["clip"]["clip_id"] == "tracknet/game9/Clip2"
                 and window.start == 0 and window.frame_step == 4)
    batch = collate_window_samples([data[index]])
    cfg = AugmentationConfig(**recipe["augmentation"])
    augmenter = VideoAugmenter(cfg, seed=recipe["seed"])
    converter = RGBToMDD(a=recipe["input_contract"]["mdd_a"], b=recipe["input_contract"]["mdd_b"])
    args.output.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    evidence: dict[str, Any] = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "run_config_sha256": digest(args.run / "config.json"),
        "manifest_sha256": digest(manifest), "augmentation": asdict(cfg),
        "clip_id": batch["clip_id"][0], "split": "val", "start": 0, "frame_step": 4,
        "frame_indices": batch["frame_indices"][0].tolist(), "timestamps_seconds": batch["timestamps"][0].tolist(),
        "image_shape": list(batch["rgb"].shape), "seed": recipe["seed"], "epoch": 0,
        "decoder": "opencv on CPU (training used nvJPEG; no pixel-identical replay claimed)",
        "profiles": {}, "display": {"mdd_scale": [0, 1], "mdd_rgb_channels": ["M-", "M+", "zero"], "gif_fps": 4},
        "interpretation": "Stress profiles force application; each profile has its own seeded camera/occluder. Input illustration only; no inference or new performance evaluation.",
    }
    with torch.inference_mode():
        for profile in PROFILES:
            rgb, labels, audit = augmenter(batch["rgb"], batch, epoch=0, profile=profile)
            mdd = converter(rgb)
            assert rgb.dtype == torch.uint8 and rgb.shape == batch["rgb"].shape
            assert mdd.dtype == torch.float32 and mdd.shape == (1, 2, 32, 720, 1280)
            assert torch.isfinite(mdd).all() and mdd.min() >= 0 and mdd.max() <= 1
            assert torch.count_nonzero(mdd[:, :, 0]) == 0
            if profile in {"clean", "occlusion"}:
                assert torch.equal(labels["uv"], batch["uv"])
                assert torch.equal(labels["position_valid"], batch["position_valid"])
            if profile == "clean":
                assert torch.equal(rgb, batch["rgb"])
            else:
                assert not audit["camera_rejected"].any()
                points = torch.cat((batch["uv"], torch.ones_like(batch["uv"][..., :1])), -1)
                expected = torch.einsum("btij,btj->bti", audit["matrix"], points)[..., :2]
                assert torch.allclose(labels["uv"], expected, atol=1e-6)
            hidden = audit.get("artificially_occluded", torch.zeros((1, 32), dtype=torch.bool))[0].numpy()
            cases[profile] = {"rgb": rgb[0].permute(0, 2, 3, 1).numpy().copy(), "mdd": mdd_image(mdd[0]), "hidden": hidden}
            evidence["profiles"][profile] = {
                "audit": {key: value.tolist() for key, value in audit.items()},
                "position_valid": labels["position_valid"][0].tolist(), "uv": labels["uv"][0].tolist(),
                "occluded_gt_frames": np.flatnonzero(hidden).tolist(),
                "rgb_sha256": hashlib.sha256(rgb.numpy()).hexdigest(),
                "mdd_sha256": hashlib.sha256(mdd.numpy()).hexdigest(),
                "mdd_mean_by_channel": mdd.mean(dim=(0, 2, 3, 4)).tolist(),
            }
            print(profile, "GT kept", labels["position_valid"].sum().item(), "GT occluded", np.flatnonzero(hidden).tolist(), flush=True)
            del mdd, rgb
    detail, selected = occlusion_detail(cases, batch)
    evidence["detail_sample_positions"] = selected
    detail.save(args.output / "occlusion-detail.png")
    for kind in ("rgb", "mdd"):
        frames = [overview(cases, batch, t, kind) for t in range(32)]
        save_gif(frames, args.output / f"input-{kind}.gif")
        frames[selected[1]].save(args.output / f"input-{kind}.png")
        print(f"Saved {kind} comparison", flush=True)
    evidence["files"] = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                         for path in sorted(args.output.glob("*")) if path.suffix in {".gif", ".png"}}
    (args.output / "inputs.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    for name, metadata in evidence["files"].items():
        print(name, metadata["bytes"], flush=True)


if __name__ == "__main__":
    main()
