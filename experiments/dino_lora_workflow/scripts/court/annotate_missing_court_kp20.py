"""Overview:
Annotate the five missing manual keypoints for CourtKP20.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/court/annotate_missing_court_kp20.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/court/annotate_missing_court_kp20.py num_images=2 dry_run=true output_dir=outputs/tmp/court_manual_kp20_dry_run

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/court/annotate_missing_court_kp20.yaml`.
    - Existing CourtKP14 points are copied from `data/court`; keypoint 14 is computed from the four court corners.
    - Only indices 15, 16, 17, 18, and 19 are clicked by the annotator.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.schema.court import COURT_KP_NAMES, NUM_COURT_KP  # noqa: E402


SOURCE_INDICES = list(range(14))
COMPUTED_INDICES = [14]
MANUAL_INDICES = [15, 16, 17, 18, 19]
IMAGE_EXTENSIONS = ("", ".png", ".jpg", ".jpeg", ".webp")


@dataclass
class AnnotationState:
    """Mutable state for the current OpenCV annotation session."""

    queue_index: int = 0
    manual_points: dict[int, list[float]] = field(default_factory=dict)
    current_manual_position: int = 0
    last_message: str = ""

    def reset_current_image(self) -> None:
        self.manual_points = {}
        self.current_manual_position = 0

    def current_manual_index(self) -> int | None:
        if self.current_manual_position >= len(MANUAL_INDICES):
            return None
        return MANUAL_INDICES[self.current_manual_position]

    def add_point(self, x: float, y: float) -> None:
        kp_index = self.current_manual_index()
        if kp_index is None:
            self.last_message = "All manual points are already set. Press Enter or n to save."
            return
        self.manual_points[kp_index] = [float(x), float(y)]
        self.current_manual_position += 1
        if self.current_manual_index() is None:
            self.last_message = "All manual points set. Press Enter or n to save."
        else:
            next_index = self.current_manual_index()
            self.last_message = f"Set {kp_index}. Next: {next_index} {COURT_KP_NAMES[next_index]}"

    def undo(self) -> None:
        if self.current_manual_position <= 0:
            self.last_message = "No manual point to undo."
            return
        self.current_manual_position -= 1
        kp_index = MANUAL_INDICES[self.current_manual_position]
        self.manual_points.pop(kp_index, None)
        self.last_message = f"Undid {kp_index} {COURT_KP_NAMES[kp_index]}"

    def is_complete(self) -> bool:
        return all(index in self.manual_points for index in MANUAL_INDICES)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_plain_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def path_for_output(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(get_original_cwd()).resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_prepared_queue_image_path(entry: dict[str, Any], output_dir: Path) -> Path | None:
    image_value = entry.get("image")
    image_path_value = entry.get("image_path")
    original_value = entry.get("original_image_path")
    candidates: list[Path] = []
    for value, prefer_output_dir in [
        (image_path_value, False),
        (image_value, True),
        (original_value, False),
    ]:
        if value is None:
            continue
        raw = Path(str(value))
        if raw.is_absolute():
            candidates.append(raw)
            continue
        if prefer_output_dir:
            candidates.append(output_dir / raw)
        candidates.append(Path(to_absolute_path(str(raw))))
        if not prefer_output_dir:
            candidates.append(output_dir / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def normalize_queue_row(entry: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    normalized = dict(entry)
    if "id" not in normalized:
        raise ValueError("Queue row is missing required field: id")
    kps14 = normalize_kps14(normalized.get("kps14"))
    if kps14 is None:
        raise ValueError(f"Queue row {normalized.get('id')} has invalid kps14")
    normalized["kps14"] = kps14

    image_path = resolve_prepared_queue_image_path(normalized, output_dir)
    if image_path is None:
        raise FileNotFoundError(f"Could not resolve queue image for {normalized.get('id')}")
    normalized["image_path"] = path_for_output(image_path)
    normalized.setdefault("original_image_path", normalized["image_path"])
    normalized.setdefault("source_indices", SOURCE_INDICES)
    normalized.setdefault("computed_indices", COMPUTED_INDICES)
    normalized.setdefault("manual_indices", MANUAL_INDICES)

    width = normalized.get("width")
    height = normalized.get("height")
    if width is None or height is None:
        width, height = read_image_size(image_path)
    normalized["width"] = int(width)
    normalized["height"] = int(height)
    return normalized


def normalize_queue_rows(queue_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    return [normalize_queue_row(row, output_dir) for row in queue_rows]


def resolve_image_path(images_dir: Path, image_id: str) -> Path | None:
    raw_id = Path(image_id)
    candidates: list[Path] = []
    if raw_id.suffix:
        candidates.append(images_dir / raw_id.name)
    else:
        for extension in IMAGE_EXTENSIONS:
            if extension:
                candidates.append(images_dir / f"{image_id}{extension}")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def normalize_kps14(kps: Any) -> list[list[float]] | None:
    if not isinstance(kps, list) or len(kps) < 14:
        return None
    normalized: list[list[float]] = []
    for point in kps[:14]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        x, y = point[0], point[1]
        if x is None or y is None:
            return None
        try:
            normalized.append([float(x), float(y)])
        except (TypeError, ValueError):
            return None
    return normalized


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def load_court_entries(input_jsons: list[str], images_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for input_json in input_jsons:
        source_path = Path(to_absolute_path(input_json))
        data = load_json(source_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {source_path}")
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                continue
            image_id = str(item["id"])
            image_path = resolve_image_path(images_dir, image_id)
            kps14 = normalize_kps14(item.get("kps"))
            if image_path is None or kps14 is None:
                continue
            entries.append(
                {
                    "id": image_id,
                    "image_path": path_for_output(image_path),
                    "source_json": path_for_output(source_path),
                    "metric": item.get("metric"),
                    "kps14": kps14,
                }
            )
    entries.sort(key=lambda row: row["id"])
    return entries


def make_queue(cfg: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    queue_path = output_dir / str(cfg.queue_file)
    if queue_path.exists():
        return normalize_queue_rows(load_jsonl(queue_path), output_dir)

    images_dir = Path(to_absolute_path(str(cfg.images_dir))).resolve()
    input_jsons = [str(path) for path in cfg.input_jsons]
    entries = load_court_entries(input_jsons, images_dir)
    if not entries:
        raise ValueError("No valid court entries found for annotation queue.")

    num_images = int(cfg.num_images)
    sample_size = len(entries) if num_images <= 0 else min(num_images, len(entries))
    rng = random.Random(int(cfg.seed))
    selected = rng.sample(entries, sample_size)
    selected.sort(key=lambda row: row["id"])

    queue_rows: list[dict[str, Any]] = []
    for item in selected:
        image_path = Path(to_absolute_path(item["image_path"]))
        width, height = read_image_size(image_path)
        queue_rows.append(
            {
                **item,
                "width": width,
                "height": height,
                "queue_created_at": now_iso(),
            }
        )

    write_jsonl(queue_path, queue_rows)
    write_json(
        output_dir / "queue_manifest.json",
        {
            "created_at": now_iso(),
            "input_jsons": input_jsons,
            "images_dir": str(cfg.images_dir),
            "queue_file": str(cfg.queue_file),
            "num_available": len(entries),
            "num_selected": len(queue_rows),
            "seed": int(cfg.seed),
        },
    )
    return normalize_queue_rows(queue_rows, output_dir)


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def compute_net_center(kps14: list[list[float]]) -> tuple[list[float], str]:
    p0 = np.asarray(kps14[0], dtype=np.float64)
    p1 = np.asarray(kps14[1], dtype=np.float64)
    p2 = np.asarray(kps14[2], dtype=np.float64)
    p3 = np.asarray(kps14[3], dtype=np.float64)

    r = p3 - p0
    s = p2 - p1
    denominator = cross2d(r, s)
    if abs(denominator) > 1e-6:
        t = cross2d(p1 - p0, s) / denominator
        intersection = p0 + t * r
        if np.all(np.isfinite(intersection)):
            return [float(intersection[0]), float(intersection[1])], "diagonal_intersection"

    mean_point = np.mean(np.stack([p0, p1, p2, p3], axis=0), axis=0)
    return [float(mean_point[0]), float(mean_point[1])], "corner_mean_fallback"


def build_kps20(
    kps14: list[list[float]],
    manual_points: dict[int, list[float]] | None = None,
) -> tuple[list[list[float] | None], str]:
    net_center, method = compute_net_center(kps14)
    kps20: list[list[float] | None] = [None for _ in range(NUM_COURT_KP)]
    for index in SOURCE_INDICES:
        kps20[index] = [float(kps14[index][0]), float(kps14[index][1])]
    kps20[14] = net_center
    if manual_points:
        for index in MANUAL_INDICES:
            if index in manual_points:
                point = manual_points[index]
                kps20[index] = [float(point[0]), float(point[1])]
    return kps20, method


def completed_ids(annotations_path: Path) -> set[str]:
    ids: set[str] = set()
    for row in load_jsonl(annotations_path):
        if "id" in row:
            ids.add(str(row["id"]))
    return ids


def build_annotation_record(entry: dict[str, Any], manual_points: dict[int, list[float]]) -> dict[str, Any]:
    kps20, method = build_kps20(entry["kps14"], manual_points)
    missing = [index for index, point in enumerate(kps20) if point is None]
    if missing:
        raise ValueError(f"Cannot save incomplete annotation for {entry['id']}: missing {missing}")
    return {
        "id": entry["id"],
        "image_path": entry["image_path"],
        "width": int(entry["width"]),
        "height": int(entry["height"]),
        "schema_name": "court_kp20",
        "kp_names": list(COURT_KP_NAMES),
        "kps20": kps20,
        "source_indices": SOURCE_INDICES,
        "computed_indices": COMPUTED_INDICES,
        "manual_indices": MANUAL_INDICES,
        "computed_metadata": {
            "14": {
                "name": COURT_KP_NAMES[14],
                "method": method,
                "source_corner_indices": [0, 1, 2, 3],
            }
        },
        "source": {
            "source_json": entry.get("source_json"),
            "metric": entry.get("metric"),
        },
        "created_at": now_iso(),
    }


def write_manifest(
    cfg: DictConfig,
    output_dir: Path,
    queue_rows: list[dict[str, Any]],
    annotations_path: Path,
    skipped_path: Path,
) -> None:
    manifest = {
        "created_at": now_iso(),
        "schema_name": "court_kp20",
        "kp_names": list(COURT_KP_NAMES),
        "source_indices": SOURCE_INDICES,
        "computed_indices": COMPUTED_INDICES,
        "manual_indices": MANUAL_INDICES,
        "queue_count": len(queue_rows),
        "completed_count": len(completed_ids(annotations_path)),
        "skipped_count": len(load_jsonl(skipped_path)),
        "files": {
            "queue": str(cfg.queue_file),
            "annotations": str(cfg.annotations_file),
            "skipped": str(cfg.skipped_file),
            "preview": str(cfg.preview_file),
        },
        "config": {
            "input_jsons": [str(path) for path in cfg.input_jsons],
            "images_dir": str(cfg.images_dir),
            "output_dir": str(cfg.output_dir),
            "num_images": int(cfg.num_images),
            "seed": int(cfg.seed),
            "dry_run": bool(cfg.dry_run),
        },
    }
    write_json(output_dir / str(cfg.manifest_file), manifest)


def write_dry_run_preview(
    cfg: DictConfig,
    output_dir: Path,
    queue_rows: list[dict[str, Any]],
) -> None:
    preview_rows: list[dict[str, Any]] = []
    methods: dict[str, int] = {}
    for entry in queue_rows:
        kps20, method = build_kps20(entry["kps14"], manual_points=None)
        methods[method] = methods.get(method, 0) + 1
        preview_rows.append(
            {
                "id": entry["id"],
                "image_path": entry["image_path"],
                "width": int(entry["width"]),
                "height": int(entry["height"]),
                "schema_name": "court_kp20",
                "kps20_preview": kps20,
                "source_indices": SOURCE_INDICES,
                "computed_indices": COMPUTED_INDICES,
                "manual_indices": MANUAL_INDICES,
                "computed_method": method,
            }
        )
    write_jsonl(output_dir / str(cfg.preview_file), preview_rows)
    write_json(
        output_dir / "dry_run_summary.json",
        {
            "created_at": now_iso(),
            "queue_count": len(queue_rows),
            "preview_file": str(cfg.preview_file),
            "computed_method_counts": methods,
            "first_ids": [row["id"] for row in queue_rows[:10]],
        },
    )


def display_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    return float(scale)


def point_to_display(point: list[float], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_annotation_view(
    entry: dict[str, Any],
    state: AnnotationState,
    cfg: DictConfig,
    completed_count: int,
    total_count: int,
) -> tuple[np.ndarray, float]:
    image_path = Path(to_absolute_path(entry["image_path"]))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    scale = display_scale(
        width,
        height,
        int(cfg.display.max_width),
        int(cfg.display.max_height),
    )
    if not math.isclose(scale, 1.0):
        image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))))

    font_scale = float(cfg.display.font_scale)
    thickness = int(cfg.display.line_thickness)
    existing_radius = int(cfg.display.existing_radius)
    computed_radius = int(cfg.display.computed_radius)
    manual_radius = int(cfg.display.manual_radius)

    kps20, method = build_kps20(entry["kps14"], state.manual_points)

    for index in SOURCE_INDICES:
        point = kps20[index]
        if point is None:
            continue
        x, y = point_to_display(point, scale)
        cv2.circle(image, (x, y), existing_radius, (0, 220, 0), -1)
        draw_label(image, str(index), (x + 4, y - 4), (0, 220, 0), font_scale, thickness)

    point14 = kps20[14]
    if point14 is not None:
        x, y = point_to_display(point14, scale)
        cv2.circle(image, (x, y), computed_radius, (255, 220, 0), 1)
        draw_label(image, "14", (x + 5, y - 5), (255, 220, 0), font_scale, thickness)

    for index in MANUAL_INDICES:
        point = kps20[index]
        color = (0, 165, 255)
        if index == state.current_manual_index():
            color = (0, 0, 255)
        if point is not None:
            x, y = point_to_display(point, scale)
            cv2.circle(image, (x, y), manual_radius, color, -1)
            draw_label(image, str(index), (x + 5, y - 5), color, font_scale, thickness)

    panel_lines = [
        f"{entry['id']}  {completed_count}/{total_count} complete",
        f"14 {COURT_KP_NAMES[14]}: {method}",
        "Enter/n: save next | u/backspace: undo | p: previous | s: skip | r: reset | q/esc: quit",
    ]
    current_index = state.current_manual_index()
    if current_index is None:
        panel_lines.append("Current: complete, press Enter/n to save")
    else:
        panel_lines.append(f"Current: {current_index} {COURT_KP_NAMES[current_index]}")
    if state.last_message:
        panel_lines.append(state.last_message)

    y = 20
    for line in panel_lines:
        draw_label(image, line, (10, y), (255, 255, 255), 0.5, 1)
        y += 20

    return image, scale


def find_next_unfinished(queue_rows: list[dict[str, Any]], start: int, done_ids: set[str]) -> int:
    index = max(start, 0)
    while index < len(queue_rows) and str(queue_rows[index]["id"]) in done_ids:
        index += 1
    return index


def run_ui(cfg: DictConfig, output_dir: Path, queue_rows: list[dict[str, Any]]) -> None:
    annotations_path = output_dir / str(cfg.annotations_file)
    skipped_path = output_dir / str(cfg.skipped_file)
    done_ids = completed_ids(annotations_path)

    state = AnnotationState()
    state.queue_index = find_next_unfinished(queue_rows, 0, done_ids)
    window_name = str(cfg.display.window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    click_scale = 1.0

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        nonlocal click_scale
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state.queue_index >= len(queue_rows):
            return
        entry = queue_rows[state.queue_index]
        original_x = min(max(x / click_scale, 0.0), float(entry["width"] - 1))
        original_y = min(max(y / click_scale, 0.0), float(entry["height"] - 1))
        state.add_point(original_x, original_y)

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        state.queue_index = find_next_unfinished(queue_rows, state.queue_index, done_ids)
        if state.queue_index >= len(queue_rows):
            print("All queued images are complete.")
            break

        entry = queue_rows[state.queue_index]
        completed_count = len(done_ids)
        view, click_scale = render_annotation_view(
            entry,
            state,
            cfg,
            completed_count=completed_count,
            total_count=len(queue_rows),
        )
        cv2.imshow(window_name, view)
        key = cv2.waitKey(30) & 0xFF

        if key in (255,):
            continue
        if key in (ord("q"), 27):
            break
        if key in (13, ord("n")):
            if not state.is_complete():
                state.last_message = "Cannot save: annotate all five manual points first."
                continue
            if str(entry["id"]) in done_ids:
                state.last_message = f"{entry['id']} is already complete."
                continue
            record = build_annotation_record(entry, state.manual_points)
            append_jsonl(annotations_path, record)
            done_ids.add(str(entry["id"]))
            print(f"Saved {entry['id']} ({len(done_ids)}/{len(queue_rows)})")
            state.queue_index += 1
            state.reset_current_image()
        elif key in (ord("u"), 8):
            state.undo()
        elif key == ord("p"):
            state.queue_index = max(state.queue_index - 1, 0)
            state.reset_current_image()
            state.last_message = "Moved to previous image."
        elif key == ord("s"):
            append_jsonl(
                skipped_path,
                {
                    "id": entry["id"],
                    "image_path": entry["image_path"],
                    "skipped_at": now_iso(),
                },
            )
            print(f"Skipped {entry['id']}")
            state.queue_index += 1
            state.reset_current_image()
        elif key == ord("r"):
            state.reset_current_image()
            state.last_message = "Reset current image annotations."

    cv2.destroyWindow(window_name)
    write_manifest(cfg, output_dir, queue_rows, annotations_path, skipped_path)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/court",
    config_name="annotate_missing_court_kp20",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_rows = make_queue(cfg, output_dir)
    annotations_path = output_dir / str(cfg.annotations_file)
    skipped_path = output_dir / str(cfg.skipped_file)

    if bool(cfg.dry_run):
        write_dry_run_preview(cfg, output_dir, queue_rows)
        write_manifest(cfg, output_dir, queue_rows, annotations_path, skipped_path)
        print(f"Dry run complete: {len(queue_rows)} queued images")
        print(f"Output: {output_dir}")
        return

    print("Controls: left-click points 15,16,17,18,19 in order.")
    print("Enter/n save-next, u/backspace undo, p previous, s skip, r reset, q/esc quit.")
    run_ui(cfg, output_dir, queue_rows)


if __name__ == "__main__":
    main()
