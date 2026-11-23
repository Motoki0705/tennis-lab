"""Copy the adopted #2363 inputs into an independent, checksummed dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PER_CAMERA = {
    "person_detection",
    "person_tracking",
    "player_selection",
    "pose_estimation",
}
PER_CLIP = {
    "court_calibration",
    "court_side",
    "camera_alignment",
    "player_association",
    "player_triangulation",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(run: Path, raw_root: Path, destination: Path) -> None:
    source = run / "inputs/dataset"
    records = json.loads((run / "input.json").read_bytes())["clips"]
    (progress,) = (run / "all/progress").glob("*/progress.json")
    expected_media: dict[Path, str] = {}
    raw_paths: set[Path] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".freeze-"
    ) as temporary:
        root = Path(temporary) / "snapshot"
        data = root / "dataset"
        data.mkdir(parents=True)
        shutil.copy2(source / "dataset.json", data)
        shutil.copy2(run / "input.json", root)
        shutil.copy2(progress, root / "progress.json")
        for record in records:
            relative = Path(record["path"])
            source_clip, target_clip = source / relative, data / relative
            store = Path("annotations/tennis_scene")
            (target_clip / store).mkdir(parents=True)
            shutil.copy2(source_clip / "clip.json", target_clip)
            clip = json.loads((target_clip / "clip.json").read_bytes())
            index_path = source_clip / store / "scene.json"
            shutil.copy2(index_path, target_clip / store)
            index = json.loads(index_path.read_bytes())
            if (
                clip["clip_id"] != record["clip_id"]
                or clip["clip_id"] != index["source"]["clip_id"]
            ):
                raise ValueError(f"Clip identity mismatch: {source_clip}")
            (target_clip / "media").mkdir()
            for camera, video in zip(
                clip["cameras"], index["source"]["videos"], strict=True
            ):
                if camera["camera_id"] != video["camera_id"]:
                    raise ValueError(f"Camera identity mismatch: {source_clip}")
                raw = Path(camera["source_path"])
                raw_paths.add(raw.relative_to(raw_root))
                media = Path(camera["video"])
                subprocess.run(
                    ["cp", "--reflink=auto", source_clip / media, target_clip / media],
                    check=True,
                )
                expected_media[target_clip / media] = video["sha256"]
            for node, reference in index["artifacts"].items():
                if node.split("/", 1)[0] in PER_CAMERA | PER_CLIP:
                    folder = Path(reference["path"]).parent
                    (target_clip / store / folder).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    subprocess.run(
                        [
                            "cp",
                            "-a",
                            "--reflink=auto",
                            source_clip / store / folder,
                            target_clip / store / folder,
                        ],
                        check=True,
                    )
        paths = sorted(p for p in root.rglob("*") if p.is_file())
        ordered_raw = sorted(raw_paths)
        with ThreadPoolExecutor(max_workers=4) as pool:
            # Use the same four source recordings for all clips; retain their whole-file hashes.
            raw_hashes = dict(
                zip(
                    (p.as_posix() for p in ordered_raw),
                    pool.map(sha256, (raw_root / p for p in ordered_raw)),
                    strict=True,
                )
            )
            raw_receipt = root / "raw-videos.json"
            raw_receipt.write_text(json.dumps(raw_hashes, indent=2) + "\n")
            paths = sorted([*paths, raw_receipt])
            with (root / "SHA256SUMS").open("w") as manifest:
                for path, digest in zip(paths, pool.map(sha256, paths), strict=True):
                    if path in expected_media and expected_media[path] != digest:
                        raise ValueError(
                            f"Synchronized media differs from scene input: {path}"
                        )
                    manifest.write(f"{digest}  {path.relative_to(root).as_posix()}\n")
        root.rename(destination)
    print(f"{sha256(destination / 'SHA256SUMS')}  {destination / 'SHA256SUMS'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    freeze(args.run, args.raw_root.resolve(), args.destination)
