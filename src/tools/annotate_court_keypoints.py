"""Manually annotate court keypoints on tennis images.

This tool provides a GUI for annotating the 20 court keypoints defined in
`src/utils/geometry/court.court_keypoints_3d()`.

Keypoint indices (CourtKP20 specification):
    0..3:  far/near doubles corners
    4..7:  far/near singles corners
    8..11: service line endpoints
    12,13: service T (far, near)
    14:    net center (ground)
    15..18: net posts (base/top, left/right)
    19:    center strap top

Example commands:
    # Annotate a single image
    uv run python -m src.tools.annotate_court_keypoints \
        input_path=data/raw/court_image.jpg \
        output.output_dir=data/court_keypoints

    # Annotate all images in a directory
    uv run python -m src.tools.annotate_court_keypoints \
        input_path=data/raw/court_images/ \
        output.output_dir=data/court_keypoints

Config entry point: `src/tools/configs/annotate_court_keypoints.yaml`
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

NUM_KEYPOINTS = 20

KEYPOINT_NAMES = [
    "far_doubles_corner_left",      # 0
    "far_doubles_corner_right",     # 1
    "near_doubles_corner_left",     # 2
    "near_doubles_corner_right",    # 3
    "far_singles_corner_left",      # 4
    "near_singles_corner_left",     # 5
    "far_singles_corner_right",     # 6
    "near_singles_corner_right",    # 7
    "far_service_left",             # 8
    "far_service_right",            # 9
    "near_service_left",            # 10
    "near_service_right",           # 11
    "far_service_T",                # 12
    "near_service_T",               # 13
    "net_center",                   # 14
    "net_post_left_base",           # 15
    "net_post_left_top",            # 16
    "net_post_right_base",          # 17
    "net_post_right_top",           # 18
    "center_strap_top",             # 19
]

COURT_LINE_CONNECTIONS = [
    # Doubles sidelines
    (0, 2),   # far left to near left doubles
    (1, 3),   # far right to near right doubles
    # Baselines (doubles)
    (0, 1),   # far baseline
    (2, 3),   # near baseline
    # Singles sidelines
    (4, 5),   # left singles
    (6, 7),   # right singles
    # Service lines
    (8, 9),   # far service line
    (10, 11), # near service line
    # Center service line
    (12, 13), # service T far to near
    # Net
    (15, 17), # net posts base
    (16, 18), # net posts top
    (14, 19), # net center to strap top
]


@dataclass
class UIConfig:
    """Configuration for annotation UI."""

    window_name: str = "Court Keypoints Annotator"
    point_radius: int = 5
    point_color_bgr: tuple[int, int, int] = (0, 255, 0)
    point_color_occluded_bgr: tuple[int, int, int] = (0, 165, 255)
    line_color_bgr: tuple[int, int, int] = (255, 255, 0)
    text_color_bgr: tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.5
    line_thickness: int = 1
    display_scale: float = 1.0
    show_court_lines: bool = True
    show_keypoint_ids: bool = True


@dataclass
class OutputConfig:
    """Configuration for output."""

    output_dir: str = "data/court_keypoints"
    annotation_format: str = "{stem}_keypoints.json"
    overwrite: bool = False


@dataclass
class ToolConfig:
    """Top-level configuration for the annotation tool."""

    mode: Literal["image", "video"] = "image"
    input_path: str | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    ui: UIConfig = field(default_factory=UIConfig)


@dataclass
class KeypointAnnotation:
    """Single keypoint annotation."""

    x: float
    y: float
    visibility: int  # 0: not labeled, 1: visible, 2: occluded

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "visibility": self.visibility}

    @classmethod
    def from_dict(cls, data: dict) -> "KeypointAnnotation":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            visibility=int(data.get("visibility", 0)),
        )


def create_empty_annotations() -> list[KeypointAnnotation]:
    """Create empty annotations for all keypoints."""
    return [KeypointAnnotation(x=0.0, y=0.0, visibility=0) for _ in range(NUM_KEYPOINTS)]


def load_annotations(path: Path) -> list[KeypointAnnotation]:
    """Load annotations from JSON file."""
    if not path.exists():
        return create_empty_annotations()

    with path.open("r") as f:
        data = json.load(f)

    keypoints = data.get("keypoints", [])
    annotations = create_empty_annotations()
    for i, kp in enumerate(keypoints[:NUM_KEYPOINTS]):
        annotations[i] = KeypointAnnotation.from_dict(kp)

    return annotations


def save_annotations(
    path: Path,
    annotations: list[KeypointAnnotation],
    image_path: Path,
    image_size: tuple[int, int],
) -> None:
    """Save annotations to JSON file."""
    data = {
        "image_path": str(image_path),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "num_keypoints": NUM_KEYPOINTS,
        "keypoints": [ann.to_dict() for ann in annotations],
        "keypoint_names": KEYPOINT_NAMES,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _draw_overlay(
    frame: np.ndarray,
    annotations: list[KeypointAnnotation],
    current_idx: int,
    config: UIConfig,
) -> np.ndarray:
    """Draw annotation overlay on frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw court lines if enabled
    if config.show_court_lines:
        for i, j in COURT_LINE_CONNECTIONS:
            if annotations[i].visibility > 0 and annotations[j].visibility > 0:
                pt1 = (int(annotations[i].x), int(annotations[i].y))
                pt2 = (int(annotations[j].x), int(annotations[j].y))
                cv2.line(
                    annotated,
                    pt1,
                    pt2,
                    config.line_color_bgr,
                    config.line_thickness,
                    cv2.LINE_AA,
                )

    # Draw keypoints
    for idx, ann in enumerate(annotations):
        if ann.visibility == 0:
            continue

        x, y = int(ann.x), int(ann.y)
        color = (
            config.point_color_bgr
            if ann.visibility == 1
            else config.point_color_occluded_bgr
        )

        # Highlight current keypoint
        if idx == current_idx:
            cv2.circle(annotated, (x, y), config.point_radius + 3, (0, 0, 255), 2)

        cv2.circle(annotated, (x, y), config.point_radius, color, -1, cv2.LINE_AA)

        if config.show_keypoint_ids:
            cv2.putText(
                annotated,
                str(idx),
                (x + config.point_radius + 2, y - config.point_radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                config.text_color_bgr,
                1,
                cv2.LINE_AA,
            )

    # Draw info panel
    info_lines = [
        f"Keypoint {current_idx}: {KEYPOINT_NAMES[current_idx]}",
        f"Status: {'Visible' if annotations[current_idx].visibility == 1 else 'Occluded' if annotations[current_idx].visibility == 2 else 'Not labeled'}",
        "",
        "Controls:",
        "LMB: set visible + next | RMB: set occluded + next | C: clear",
        "N/P or UP/DOWN: next/prev keypoint",
        "0-9: jump to keypoint | S: save | Q: save+quit",
    ]

    # Draw semi-transparent background for info
    panel_h = len(info_lines) * 20 + 10
    overlay = annotated.copy()
    cv2.rectangle(overlay, (5, 5), (400, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    for i, text in enumerate(info_lines):
        cv2.putText(
            annotated,
            text,
            (10, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            config.text_color_bgr,
            1,
            cv2.LINE_AA,
        )

    # Apply display scale
    if config.display_scale != 1.0:
        annotated = cv2.resize(
            annotated,
            None,
            fx=config.display_scale,
            fy=config.display_scale,
            interpolation=cv2.INTER_AREA,
        )

    return annotated


def annotate_image(
    image_path: Path,
    output_config: OutputConfig,
    ui_config: UIConfig,
) -> None:
    """Annotate a single image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = frame.shape[:2]

    # Resolve output path
    output_dir = Path(output_config.output_dir)
    annotation_name = output_config.annotation_format.format(stem=image_path.stem)
    annotation_path = output_dir / annotation_name

    if annotation_path.exists() and not output_config.overwrite:
        annotations = load_annotations(annotation_path)
        LOGGER.info("Loaded existing annotations from %s", annotation_path)
    else:
        annotations = create_empty_annotations()

    state = {"current_idx": 0}

    def on_mouse(event: int, mx: int, my: int, _flags: int, _param: object) -> None:
        # Adjust for display scale
        x = int(mx / ui_config.display_scale)
        y = int(my / ui_config.display_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Set visible keypoint
            annotations[state["current_idx"]] = KeypointAnnotation(
                x=float(x), y=float(y), visibility=1
            )
            state["current_idx"] = (state["current_idx"] + 1) % NUM_KEYPOINTS
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Set occluded keypoint
            annotations[state["current_idx"]] = KeypointAnnotation(
                x=float(x), y=float(y), visibility=2
            )
            state["current_idx"] = (state["current_idx"] + 1) % NUM_KEYPOINTS

    cv2.namedWindow(ui_config.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ui_config.window_name, on_mouse)

    try:
        while True:
            overlay = _draw_overlay(frame, annotations, state["current_idx"], ui_config)
            cv2.imshow(ui_config.window_name, overlay)
            key = cv2.waitKey(30) & 0xFF

            if key in {ord("n"), ord(" "), 82}:  # n, space, or UP arrow
                state["current_idx"] = (state["current_idx"] + 1) % NUM_KEYPOINTS
            elif key in {ord("p"), 84}:  # p or DOWN arrow
                state["current_idx"] = (state["current_idx"] - 1) % NUM_KEYPOINTS
            elif key == ord("c"):
                # Clear current keypoint
                annotations[state["current_idx"]] = KeypointAnnotation(
                    x=0.0, y=0.0, visibility=0
                )
            elif key == ord("s"):
                # Save
                save_annotations(annotation_path, annotations, image_path, (w, h))
                LOGGER.info("Saved annotations to %s", annotation_path)
            elif ord("0") <= key <= ord("9"):
                # Jump to keypoint 0-9
                target = key - ord("0")
                if target < NUM_KEYPOINTS:
                    state["current_idx"] = target
            elif key in {ord("q"), 27}:  # q or ESC
                save_annotations(annotation_path, annotations, image_path, (w, h))
                LOGGER.info("Saved annotations to %s", annotation_path)
                break

    finally:
        cv2.destroyWindow(ui_config.window_name)


def annotate_directory(
    input_dir: Path,
    output_config: OutputConfig,
    ui_config: UIConfig,
) -> None:
    """Annotate all images in a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    LOGGER.info("Found %d images to annotate", len(image_files))

    for i, image_path in enumerate(image_files):
        LOGGER.info("Annotating image %d/%d: %s", i + 1, len(image_files), image_path.name)
        annotate_image(image_path, output_config, ui_config)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="annotate_court_keypoints",
)
def main(cfg: DictConfig) -> None:
    """Run the court keypoints annotation tool."""
    config = ToolConfig(
        mode=cfg.get("mode", "image"),
        input_path=cfg.get("input_path", None),
        output=OutputConfig(
            output_dir=cfg.output.get("output_dir", "data/court_keypoints"),
            annotation_format=cfg.output.get("annotation_format", "{stem}_keypoints.json"),
            overwrite=bool(cfg.output.get("overwrite", False)),
        ),
        ui=UIConfig(
            window_name=cfg.ui.get("window_name", "Court Keypoints Annotator"),
            point_radius=int(cfg.ui.get("point_radius", 5)),
            point_color_bgr=tuple(cfg.ui.get("point_color_bgr", [0, 255, 0])),
            point_color_occluded_bgr=tuple(cfg.ui.get("point_color_occluded_bgr", [0, 165, 255])),
            line_color_bgr=tuple(cfg.ui.get("line_color_bgr", [255, 255, 0])),
            text_color_bgr=tuple(cfg.ui.get("text_color_bgr", [255, 255, 255])),
            font_scale=float(cfg.ui.get("font_scale", 0.5)),
            line_thickness=int(cfg.ui.get("line_thickness", 1)),
            display_scale=float(cfg.ui.get("display_scale", 1.0)),
            show_court_lines=bool(cfg.ui.get("show_court_lines", True)),
            show_keypoint_ids=bool(cfg.ui.get("show_keypoint_ids", True)),
        ),
    )

    if config.input_path is None:
        raise ValueError("input_path is required")

    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_dir():
        annotate_directory(input_path, config.output, config.ui)
    else:
        annotate_image(input_path, config.output, config.ui)


if __name__ == "__main__":
    main()
