"""Record real multi-head inference without publishing any synthetic scene."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.inference.checkpoint import file_sha256
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
)


def _scene_snapshot(root: Path | None) -> dict[str, str]:
    if root is None:
        return {}
    files = []
    for scene in ("B00", "B01", "B02", "B03"):
        directory = root / scene
        for relative in (
            "alignment/alignment.json",
            "alignment/ground-line-map.npz",
            "resolved-config.yaml",
            "run-manifest.json",
        ):
            if (directory / relative).is_file():
                files.append(directory / relative)
        files.extend(directory.glob("datasets/*/dataset.json"))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "predictions", "figures"):
        (output / name).mkdir()
    before = _scene_snapshot(args.scene_root)
    start = time.monotonic()
    predictor = CourtPredictor.load_from_checkpoint(
        args.checkpoint.resolve(), device=args.device
    )
    load_seconds = time.monotonic() - start
    records = []
    for index, source in enumerate(args.image, 1):
        ident = f"{index:02d}_" + re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem)
        local_input = output / "inputs" / (ident + source.suffix)
        shutil.copy2(source, local_input)
        rgb = np.asarray(
            ImageOps.exif_transpose(Image.open(local_input)).convert("RGB")
        )
        start = time.monotonic()
        prediction = predictor.predict(rgb, heads=("kp", "line"))
        elapsed = time.monotonic() - start
        kp, line = prediction.raw_heads["kp"], prediction.raw_heads["line"]
        assert isinstance(kp, CourtKeypointPrediction) and isinstance(
            line, CourtLinePrediction
        )
        result = prediction.homography
        assert result is not None
        points, valid = prediction.downstream_keypoints()
        npz_path = output / "predictions" / f"{ident}.npz"
        np.savez_compressed(
            npz_path,
            raw_kp=kp.keypoints.numpy(),
            scores=kp.scores.numpy(),
            raw_valid=kp.valid.numpy(),
            line_probability=line.probability.numpy(),
            fitted_kp=points,
            fitted_valid=valid,
            selected=result.selected,
            homography=np.full((3, 3), np.nan)
            if result.matrix is None
            else result.matrix,
        )
        fitted = rgb.copy()
        if result.matrix is not None:
            for a, b in COURT_HOMOGRAPHY_EDGES:
                start_xy, stop_xy = np.rint(points[[a, b]]).astype(int)
                cv2.line(
                    fitted, tuple(start_xy), tuple(stop_xy), (0, 20, 20), 4, cv2.LINE_AA
                )
                cv2.line(
                    fitted,
                    tuple(start_xy),
                    tuple(stop_xy),
                    (0, 225, 195),
                    2,
                    cv2.LINE_AA,
                )
        for i, point in enumerate(kp.keypoints[:, 0].numpy()):
            if not bool(kp.valid[i, 0]):
                continue
            xy = tuple(np.rint(point).astype(int))
            if result.selected[i]:
                cv2.circle(fitted, xy, 4, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.drawMarker(
                    fitted, xy, (160, 160, 160), cv2.MARKER_TILTED_CROSS, 9, 2
                )
        mask = cv2.resize(line.probability.numpy(), (rgb.shape[1], rgb.shape[0])) >= 0.5
        line_overlay = rgb.copy()
        line_overlay[mask] = (0.2 * rgb[mask] + 0.8 * np.array([0, 225, 195])).astype(
            np.uint8
        )
        panels = Image.new("RGB", (1440, 350), "white")
        draw = ImageDraw.Draw(panels)
        for col, (array, title) in enumerate(
            (
                (rgb, "RGB"),
                (line_overlay, "raw LINE p>=0.5"),
                (
                    fitted,
                    f"hybrid: {result.status}; used {int(result.selected.sum())}/14",
                ),
            )
        ):
            panel = ImageOps.contain(Image.fromarray(array), (480, 320))
            panels.paste(
                panel,
                (col * 480 + (480 - panel.width) // 2, 25 + (320 - panel.height) // 2),
            )
            draw.text((col * 480 + 6, 6), f"{ident} / {title}", fill="black")
        figure_path = output / "figures" / f"{ident}.png"
        panels.save(figure_path)
        record = {
            "id": ident,
            "source": str(source.resolve()),
            "input": str(local_input.relative_to(output)),
            "input_sha256": file_sha256(local_input),
            "native_size_hw": list(prediction.native_size_hw),
            "inference_seconds": elapsed,
            **prediction.geometry_diagnostics(),
            "predictions": str(npz_path.relative_to(output)),
            "predictions_sha256": file_sha256(npz_path),
            "figure": str(figure_path.relative_to(output)),
            "figure_sha256": file_sha256(figure_path),
        }
        records.append(record)
        print(
            ident,
            result.status,
            "selected",
            int(result.selected.sum()),
            "seconds",
            round(elapsed, 3),
            flush=True,
        )
    after = _scene_snapshot(args.scene_root)
    if before != after:
        raise RuntimeError(
            "Scene owner metadata changed during this read-only inference audit"
        )
    code_root = Path(__file__).resolve().parents[4]
    code = [
        *sorted((code_root / "src/tasks/court_detection/inference").glob("*.py")),
        *sorted((code_root / "src/tasks/court_detection/geometry").glob("*.py")),
    ]
    report = {
        "schema": "court_hybrid_inference_audit_v1",
        "checkpoint": predictor.checkpoint_identity,
        "device": args.device,
        "load_seconds": load_seconds,
        "hybrid_config": asdict(predictor.hybrid_config),
        "source_sha256": {
            str(path.relative_to(code_root)): file_sha256(path) for path in code
        },
        "scene_owner_sha256_before_and_after": before,
        "scene_owners_unchanged": True,
        "images": records,
        "scope": "Real saved-model inference. LINE support is internal agreement, not human-GT accuracy. Geometry is in the input KP channel convention; no multiview physical identity is inferred.",
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
