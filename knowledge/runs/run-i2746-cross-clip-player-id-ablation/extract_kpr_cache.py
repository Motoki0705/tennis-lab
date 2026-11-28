"""Extract KPR (Keypoint Promptable Re-ID [P8]) native-part features on the EXACT same 4,416 crops.

Compares on the same fixed KPR Market/SOLIDER checkpoint and exact crop set:
- unprompted: zero keypoint prompts, no negative prompts
- prompted: existing 2D COCO-17 target keypoints as positive prompts + co-occurring other-person
  keypoints inside the crop as negative prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.tasks.player_association.appearance.kpr import KprEncoder
from src.tasks.player_association.appearance.sampling import crop as reid_crop

DEFAULT_BENCH_ROOT = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4")
DEFAULT_KPR_CKPT = Path("ckpt/player_association/kpr/kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar")
DEFAULT_CACHE_DIR = Path("outputs/i2746_cross_clip_player_id/cache")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_kpr(
    bench_root: Path,
    cache_dir: Path,
    ckpt_path: Path,
    device: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    clip_cache = np.load(cache_dir / "clip_reid_cache.npz", allow_pickle=False)
    clip_names = [str(x) for x in clip_cache["clip_names"].tolist()]
    crop_valid = clip_cache["crop_valid"]  # (69, 4, 4, 4)
    crop_frames = clip_cache["crop_frames"]
    crop_boxes = clip_cache["crop_boxes"]
    crop_kp_norm = clip_cache["crop_kp_norm"]  # (69, 4, 4, 4, 17, 3)
    crop_neg_kp_norm = clip_cache["crop_neg_kp_norm"]  # (69, 4, 4, 4, 3, 17, 3)

    N, P, V, K = crop_valid.shape
    encoder = KprEncoder(ckpt_path, device=device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    emb_unprompted = np.zeros((N, P, V, K, 6, 512), dtype=np.float32)
    vis_unprompted = np.zeros((N, P, V, K, 6), dtype=bool)
    emb_prompted = np.zeros((N, P, V, K, 6, 512), dtype=np.float32)
    vis_prompted = np.zeros((N, P, V, K, 6), dtype=bool)

    gpu_unprompted_sec = 0.0
    gpu_prompted_sec = 0.0
    total_crops = 0
    crops_with_neg_prompt = 0

    for ci, cname in enumerate(clip_names):
        clip_dir = bench_root / "videos/video_000/clips" / cname
        wanted_by_cam: dict[int, dict[int, list[tuple[int, int]]]] = {v: {} for v in range(V)}
        for p in range(P):
            for v in range(V):
                for ki in range(K):
                    if crop_valid[ci, p, v, ki]:
                        f = int(crop_frames[ci, p, v, ki])
                        wanted_by_cam[v].setdefault(f, []).append((p, ki))

        tensors: list[np.ndarray] = []
        pos_prompts: list[np.ndarray] = []
        neg_prompts: list[np.ndarray] = []
        coords: list[tuple[int, int, int]] = []

        for v in range(V):
            wanted = wanted_by_cam[v]
            if not wanted:
                continue
            cap = cv2.VideoCapture(str(clip_dir / f"media/cam{v}.mp4"))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {clip_dir}/media/cam{v}.mp4")
            max_f = max(wanted)
            fidx = 0
            while fidx <= max_f:
                ok, frame = cap.read()
                if not ok:
                    break
                items = wanted.get(fidx)
                if items:
                    for p, ki in items:
                        box = crop_boxes[ci, p, v, ki]
                        img_chw = reid_crop(frame, box, encoder.input_size)
                        tensors.append(img_chw)
                        pos_prompts.append(crop_kp_norm[ci, p, v, ki])
                        neg_prompts.append(crop_neg_kp_norm[ci, p, v, ki])
                        if int(clip_cache["crop_neg_count"][ci, p, v, ki]) > 0:
                            crops_with_neg_prompt += 1
                        coords.append((p, v, ki))
                fidx += 1
            cap.release()

        if not tensors:
            continue
        total_crops += len(tensors)
        zero_prompts = np.zeros((len(tensors), 17, 3), dtype=np.float32)
        pos_arr = np.stack(pos_prompts, axis=0).astype(np.float32)
        neg_arr = np.stack(neg_prompts, axis=0).astype(np.float32)

        for start in range(0, len(tensors), batch_size):
            end = min(len(tensors), start + batch_size)
            bt = torch.from_numpy(np.stack(tensors[start:end], axis=0))
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            eu, vu, _ = encoder.extract(bt, zero_prompts[start:end], negative=None)
            if use_cuda:
                torch.cuda.synchronize()
            gpu_unprompted_sec += time.perf_counter() - t1

            t2 = time.perf_counter()
            ep, vp, _ = encoder.extract(bt, pos_arr[start:end], negative=neg_arr[start:end])
            if use_cuda:
                torch.cuda.synchronize()
            gpu_prompted_sec += time.perf_counter() - t2

            eu_np, vu_np = eu.numpy(), vu.numpy()
            ep_np, vp_np = ep.numpy(), vp.numpy()
            for idx_b, (p, v, ki) in enumerate(coords[start:end]):
                emb_unprompted[ci, p, v, ki] = eu_np[idx_b]
                vis_unprompted[ci, p, v, ki] = vu_np[idx_b]
                emb_prompted[ci, p, v, ki] = ep_np[idx_b]
                vis_prompted[ci, p, v, ki] = vp_np[idx_b]

    out_npz = cache_dir / "kpr_cache.npz"
    np.savez_compressed(
        out_npz,
        clip_names=np.asarray(clip_names),
        emb_unprompted=emb_unprompted,
        vis_unprompted=vis_unprompted,
        emb_prompted=emb_prompted,
        vis_prompted=vis_prompted,
    )
    wall_sec = time.perf_counter() - t0_wall
    cpu_sec = time.process_time() - t0_cpu
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024.0**2)) if use_cuda else 0.0

    manifest = {
        "schema_version": 1,
        "encoder": {
            "name": encoder.name,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": file_sha256(ckpt_path),
            "input_size_hw": list(encoder.input_size),
            "num_parts": 6,
            "part_dim": 512,
        },
        "source_clip_reid_cache_sha256": file_sha256(cache_dir / "clip_reid_cache.npz"),
        "stats": {
            "total_crops": total_crops,
            "crops_with_negative_prompt": crops_with_neg_prompt,
            "wall_sec": round(wall_sec, 3),
            "cpu_sec": round(cpu_sec, 3),
            "gpu_unprompted_sec": round(gpu_unprompted_sec, 3),
            "gpu_prompted_sec": round(gpu_prompted_sec, 3),
            "peak_rss_mb": round(peak_rss_mb, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "cache_npz_sha256": file_sha256(out_npz),
            "cache_npz_bytes": out_npz.stat().st_size,
            "device": device,
        },
    }
    (cache_dir / "kpr_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"Extracted KPR (unprompted + prompted) for {total_crops} crops ({crops_with_neg_prompt} with neg prompt) "
        f"in {wall_sec:.2f}s (gpu_unp={gpu_unprompted_sec:.2f}s, gpu_prm={gpu_prompted_sec:.2f}s, vram={peak_vram_mb:.1f}MB)"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_KPR_CKPT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    extract_kpr(args.bench_root, args.cache_dir, args.ckpt, args.device, args.batch_size)


if __name__ == "__main__":
    main()
