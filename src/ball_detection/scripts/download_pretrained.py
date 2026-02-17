"""Download pretrained checkpoints for ball_detection fine-tuning."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve


WASB_TENNIS_FILE_ID = "14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z"
TRACKNETV3_CKPTS_ZIP_FILE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"


def _drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


def _download_file(url: str, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_google_drive = "drive.google.com" in url or "drive.usercontent.google.com" in url
    if is_google_drive:
        python_bin = Path(__file__).resolve().parents[3] / ".venv" / "bin" / "python"
        gdown_bin = shutil.which("gdown")
        if gdown_bin is not None:
            subprocess.run([gdown_bin, "--fuzzy", url, "-O", str(output_path)], check=True)
            return

        parsed = urlparse(url)
        file_id = parse_qs(parsed.query).get("id", [None])[0]
        if file_id is not None and python_bin.exists():
            result = subprocess.run(
                [str(python_bin), "-m", "gdown", file_id, "-O", str(output_path)],
                check=False,
            )
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return

    wget_bin = shutil.which("wget")
    if wget_bin is not None:
        subprocess.run([wget_bin, url, "-O", str(output_path)], check=True)
        return

    urlretrieve(url, str(output_path))


def _extract_tracknet_weights(zip_path: Path, target_dir: Path) -> None:
    extract_dir = target_dir / "_tmp_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    for filename in ("TrackNet_best.pt", "InpaintNet_best.pt"):
        candidates = list(extract_dir.rglob(filename))
        if not candidates:
            raise FileNotFoundError(f"{filename} was not found in archive: {zip_path}")
        target_path = target_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], target_path)

    shutil.rmtree(extract_dir)


def _download_wasb_tennis(root_dir: Path, *, overwrite: bool) -> Path:
    output_path = root_dir / "checkpoints" / "wasb" / "wasb_tennis_best.pth.tar"
    _download_file(_drive_url(WASB_TENNIS_FILE_ID), output_path, overwrite=overwrite)
    return output_path


def _download_tracknetv3(root_dir: Path, *, overwrite: bool, keep_archive: bool) -> tuple[Path, Path]:
    target_dir = root_dir / "checkpoints" / "tracknetv3"
    archive_path = target_dir / "TrackNetV3_ckpts.zip"
    _download_file(_drive_url(TRACKNETV3_CKPTS_ZIP_FILE_ID), archive_path, overwrite=overwrite)
    _extract_tracknet_weights(archive_path, target_dir)

    if not keep_archive and archive_path.exists():
        archive_path.unlink()

    return target_dir / "TrackNet_best.pt", target_dir / "InpaintNet_best.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained checkpoints for ball_detection.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["wasb", "tracknetv3", "all"],
        default=["all"],
        help="Which checkpoints to download.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep TrackNetV3 zip archive after extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root_dir = args.root_dir.resolve()

    selected = set(args.models)
    if "all" in selected:
        selected = {"wasb", "tracknetv3"}

    if "wasb" in selected:
        path = _download_wasb_tennis(root_dir, overwrite=bool(args.overwrite))
        print(f"Downloaded WASB tennis checkpoint: {path}")

    if "tracknetv3" in selected:
        tracknet_path, inpaint_path = _download_tracknetv3(
            root_dir,
            overwrite=bool(args.overwrite),
            keep_archive=bool(args.keep_archive),
        )
        print(f"Downloaded TrackNetV3 checkpoint: {tracknet_path}")
        print(f"Downloaded InpaintNet checkpoint: {inpaint_path}")


if __name__ == "__main__":
    main()
