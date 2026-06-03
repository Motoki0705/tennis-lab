"""Overview:
Prepare a fixed CourtKP20 manual seed queue from existing CourtKP14 data.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/court/prepare_manual_court_kp20_seed.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/court/prepare_manual_court_kp20_seed.py num_images=2 dry_run=true
    .venv/bin/python experiments/dino_lora_workflow/scripts/court/prepare_manual_court_kp20_seed.py num_images=2 output_dir=outputs/tmp/court_manual_seed

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/court/prepare_manual_court_kp20_seed.yaml`.
    - The queue keeps CourtKP14 points at indices 0..13, marks index 14 as computed, and asks the UI to annotate only indices 15..19.
    - If `reuse_existing_queue=true` and the queue already exists, the script leaves the prepared seed set unchanged.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "utils" / "schema" / "court.py").is_file():
            return candidate
    return start.parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.schema.court import COURT_KP_NAMES  # noqa: E402


SOURCE_INDICES = list(range(14))
COMPUTED_INDICES = [14]
MANUAL_INDICES = [15, 16, 17, 18, 19]
IMAGE_EXTENSIONS = ("", ".png", ".jpg", ".jpeg", ".webp", ".bmp")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_plain_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def path_for_output(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(get_original_cwd()).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_paths(values: Any, *, field_name: str) -> list[Path]:
    paths = [Path(to_absolute_path(str(value))).resolve() for value in values]
    if not paths:
        raise ValueError(f"{field_name} must contain at least one path")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{field_name} contains missing files: {missing}")
    return paths


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "image"


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def normalize_kps14(kps: Any) -> list[list[float]] | None:
    if not isinstance(kps, list) or len(kps) < 14:
        return None
    points: list[list[float]] = []
    for point in kps[:14]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        x = numeric(point[0])
        y = numeric(point[1])
        if x is None or y is None:
            return None
        points.append([float(x), float(y)])
    return points


def resolve_image_path(images_dir: Path, image_id: str) -> Path | None:
    raw = Path(image_id)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    if raw.suffix:
        candidates.extend([images_dir / raw, images_dir / raw.name])
    else:
        for extension in IMAGE_EXTENSIONS:
            candidates.append(images_dir / f"{image_id}{extension}" if extension else images_dir / image_id)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def collect_candidates(input_jsons: list[Path], images_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates_by_id: dict[str, dict[str, Any]] = {}
    stats = {
        "input_rows": 0,
        "duplicate_ids": 0,
        "missing_images": 0,
        "invalid_kps14": 0,
    }

    for input_json in input_jsons:
        data = load_json(input_json)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of court rows in {input_json}")
        for row_index, item in enumerate(data, start=1):
            stats["input_rows"] += 1
            if not isinstance(item, dict) or item.get("id") is None:
                stats["invalid_kps14"] += 1
                continue
            image_id = str(item["id"])
            if image_id in candidates_by_id:
                stats["duplicate_ids"] += 1
                continue
            kps14 = normalize_kps14(item.get("kps"))
            if kps14 is None:
                stats["invalid_kps14"] += 1
                continue
            image_path = resolve_image_path(images_dir, image_id)
            if image_path is None:
                stats["missing_images"] += 1
                continue
            candidates_by_id[image_id] = {
                "id": image_id,
                "source_json": path_for_output(input_json),
                "source_row_index": row_index,
                "metric": item.get("metric"),
                "original_image_path": path_for_output(image_path),
                "absolute_image_path": str(image_path),
                "kps14": kps14,
            }

    return sorted(candidates_by_id.values(), key=lambda row: row["id"]), stats


def select_candidates(candidates: list[dict[str, Any]], *, num_images: int, seed: int) -> list[dict[str, Any]]:
    sample_size = len(candidates) if num_images <= 0 else min(num_images, len(candidates))
    rng = random.Random(seed)
    selected = rng.sample(candidates, sample_size)
    selected.sort(key=lambda row: row["id"])
    return selected


def destination_image_path(output_dir: Path, source_path: Path, image_id: str) -> Path:
    suffix = source_path.suffix.lower() or ".png"
    return output_dir / "images" / f"{safe_stem(image_id)}{suffix}"


def make_queue_rows(
    selected: list[dict[str, Any]],
    *,
    output_dir: Path,
    copy_images: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    created_at = now_iso()
    for item in selected:
        source_path = Path(str(item["absolute_image_path"])).resolve()
        size = read_image_size(source_path)
        if size is None:
            raise FileNotFoundError(f"Could not read selected image: {source_path}")
        width, height = size
        destination = destination_image_path(output_dir, source_path, str(item["id"]))
        if copy_images:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
        rows.append(
            {
                "id": item["id"],
                "image": destination.relative_to(output_dir).as_posix(),
                "original_image_path": item["original_image_path"],
                "width": int(width),
                "height": int(height),
                "kps14": item["kps14"],
                "manual_indices": MANUAL_INDICES,
                "computed_indices": COMPUTED_INDICES,
                "source_indices": SOURCE_INDICES,
                "schema_name": "court_kp20_seed_queue",
                "source": {
                    "source_json": item["source_json"],
                    "source_row_index": item["source_row_index"],
                    "metric": item.get("metric"),
                },
                "queue_created_at": created_at,
            }
        )
    return rows


def build_manifest(
    *,
    cfg: DictConfig,
    input_jsons: list[Path],
    images_dir: Path,
    output_dir: Path,
    queue_rows: list[dict[str, Any]],
    candidate_count: int,
    stats: dict[str, int],
    reused_existing_queue: bool,
) -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "schema_name": "court_kp20_seed_queue",
        "kp_names": list(COURT_KP_NAMES),
        "source_indices": SOURCE_INDICES,
        "computed_indices": COMPUTED_INDICES,
        "manual_indices": MANUAL_INDICES,
        "input_jsons": [path_for_output(path) for path in input_jsons],
        "images_dir": path_for_output(images_dir),
        "output_dir": path_for_output(output_dir),
        "queue_file": str(cfg.queue_file),
        "candidate_count": candidate_count,
        "selected_count": len(queue_rows),
        "selected_ids": [str(row["id"]) for row in queue_rows],
        "seed": int(cfg.seed),
        "num_images": int(cfg.num_images),
        "reused_existing_queue": reused_existing_queue,
        "stats": stats,
        "config": as_plain_container(cfg),
    }


def write_readme(path: Path, queue_file: str) -> None:
    text = f"""# CourtKP20 Manual Seed

This directory contains a fixed manual annotation queue for CourtKP20.

1. Annotate only keypoints `15..19` with:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/court/annotate_missing_court_kp20.py \\
  output_dir={path.parent.as_posix()} \\
  queue_file={queue_file}
```

2. Convert completed point annotations to Grounding DINO boxes:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py \\
  source_format=court_kp20_points_jsonl \\
  input_paths='[{path.parent.as_posix()}/annotations_points.jsonl]' \\
  output_dir=data/dino_workflow/court/manual100_kp20 \\
  label_source=manual
```

Existing CourtKP14 points are stored in the queue at indices `0..13`.
Index `14` is computed by the annotation UI from the court corners.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize(
    *,
    dry_run: bool,
    output_dir: Path,
    queue_rows: list[dict[str, Any]],
    candidate_count: int,
    stats: dict[str, int],
    reused_existing_queue: bool,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "output_dir": path_for_output(output_dir),
        "candidate_count": candidate_count,
        "selected_count": len(queue_rows),
        "selected_ids": [str(row["id"]) for row in queue_rows],
        "reused_existing_queue": reused_existing_queue,
        "stats": stats,
    }


def prepare_manual_court_kp20_seed(cfg: DictConfig) -> dict[str, Any]:
    input_jsons = resolve_paths(cfg.input_jsons, field_name="input_jsons")
    images_dir = Path(to_absolute_path(str(cfg.images_dir))).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    queue_path = output_dir / str(cfg.queue_file)
    manifest_path = output_dir / str(cfg.manifest_file)
    readme_path = output_dir / str(cfg.readme_file)

    if queue_path.exists() and bool(cfg.reuse_existing_queue) and not bool(cfg.overwrite):
        existing_rows = load_jsonl(queue_path)
        candidates, stats = collect_candidates(input_jsons, images_dir)
        manifest = build_manifest(
            cfg=cfg,
            input_jsons=input_jsons,
            images_dir=images_dir,
            output_dir=output_dir,
            queue_rows=existing_rows,
            candidate_count=len(candidates),
            stats=stats,
            reused_existing_queue=True,
        )
        if not bool(cfg.dry_run):
            write_json(manifest_path, manifest)
        return summarize(
            dry_run=bool(cfg.dry_run),
            output_dir=output_dir,
            queue_rows=existing_rows,
            candidate_count=len(candidates),
            stats=stats,
            reused_existing_queue=True,
        )

    if queue_path.exists() and not bool(cfg.overwrite) and not bool(cfg.dry_run):
        raise FileExistsError(
            f"Queue already exists: {queue_path}. Set reuse_existing_queue=true or overwrite=true."
        )

    candidates, stats = collect_candidates(input_jsons, images_dir)
    if not candidates:
        raise ValueError("No valid court KP14 candidates were found.")
    selected = select_candidates(
        candidates,
        num_images=int(cfg.num_images),
        seed=int(cfg.seed),
    )

    if bool(cfg.dry_run):
        return summarize(
            dry_run=True,
            output_dir=output_dir,
            queue_rows=[
                {
                    "id": row["id"],
                    "image": f"images/{safe_stem(str(row['id']))}{Path(str(row['absolute_image_path'])).suffix.lower() or '.png'}",
                }
                for row in selected
            ],
            candidate_count=len(candidates),
            stats=stats,
            reused_existing_queue=False,
        )

    queue_rows = make_queue_rows(
        selected,
        output_dir=output_dir,
        copy_images=bool(cfg.copy_images),
    )
    manifest = build_manifest(
        cfg=cfg,
        input_jsons=input_jsons,
        images_dir=images_dir,
        output_dir=output_dir,
        queue_rows=queue_rows,
        candidate_count=len(candidates),
        stats=stats,
        reused_existing_queue=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(queue_path, queue_rows)
    write_json(manifest_path, manifest)
    if bool(cfg.write_readme):
        write_readme(readme_path, str(cfg.queue_file))

    return summarize(
        dry_run=False,
        output_dir=output_dir,
        queue_rows=queue_rows,
        candidate_count=len(candidates),
        stats=stats,
        reused_existing_queue=False,
    )


@hydra.main(
    version_base="1.3",
    config_path="../../configs/court",
    config_name="prepare_manual_court_kp20_seed",
)
def main(cfg: DictConfig) -> None:
    summary = prepare_manual_court_kp20_seed(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
