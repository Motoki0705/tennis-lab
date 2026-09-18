"""Check supplied photos against local real and rendered training corpora."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from common import REPO, ROOT, sha256, sources, write_json
from PIL import Image, ImageOps


def signature(path: Path) -> tuple[str, str, np.ndarray]:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        low = cv2.dct(np.asarray(rgb.convert("L").resize((32, 32)), np.float32))[
            :8, :8
        ].ravel()[1:]
    return sha256(path), digest, low > np.median(low)


def main() -> None:
    records = sources()
    targets = [signature(ROOT / r["paper_path"]) for r in records]
    data = REPO / "data/court"
    train = json.loads((data / "data_train.json").read_text())
    val = json.loads((data / "data_val.json").read_text())
    corpora = {"TCD": sorted((data / "images").glob("*"))}
    manifests = {}
    for sid in ("B00", "B01", "B02", "B03"):
        p = REPO / "data/synthetic_data_generation/scenes" / sid / "datasets/court"
        manifest = p / "dataset.json"
        d = json.loads(manifest.read_text())
        corpora[sid] = [p / s["rgb_preview"] for s in d["samples"]]
        manifests[sid] = sha256(manifest)
    corpus_results = {}
    for name, paths in corpora.items():
        nearest = [{"distance": 64, "path": None} for _ in records]
        exact, pixels = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for path, (filehash, rgbhash, phash) in zip(
                paths, pool.map(signature, paths), strict=True
            ):
                for i, (tf, trgb, th) in enumerate(targets):
                    pair = {"id": records[i]["id"], "path": str(path.relative_to(REPO))}
                    if tf == filehash:
                        exact.append(pair)
                    if trgb == rgbhash:
                        pixels.append(pair)
                    distance = int(np.count_nonzero(phash != th))
                    if distance < nearest[i]["distance"]:
                        nearest[i] = {"distance": distance, "path": pair["path"]}
        corpus_results[name] = {
            "count": len(paths),
            "file_matches": exact,
            "pixel_matches": pixels,
            "nearest_phash63": {
                r["id"]: n for r, n in zip(records, nearest, strict=True)
            },
        }
        print(
            f"Audited {name}: {len(paths)} images, {len(exact)} byte and {len(pixels)} pixel matches",
            flush=True,
        )
    tr = {r["id"].rsplit("_", 1)[0] for r in train}
    va = {r["id"].rsplit("_", 1)[0] for r in val}
    write_json(
        ROOT / "evidence/dataset_audit.json",
        {
            "input_manifest_sha256": sha256(ROOT / "evidence/inputs.json"),
            "corpora": corpus_results,
            "synthetic_manifest_sha256": manifests,
            "tcd_splits": {
                "train_rows": len(train),
                "val_rows": len(val),
                "train_video_ids": len(tr),
                "val_video_ids": len(va),
                "shared_video_ids": len(tr & va),
            },
            "scope": "All local TCD images and B00-B03 rendered previews across train/validation/test; no unreadable file is skipped.",
            "limitation": "No exact match is not proof of unseen venues or absence from inaccessible DINOv3 pretraining. Perceptual hashes screen similarity, not independence.",
        },
    )


if __name__ == "__main__":
    main()
