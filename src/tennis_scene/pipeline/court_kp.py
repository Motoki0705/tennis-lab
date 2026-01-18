"""Court keypoint detection module."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from src.tennis_scene.pipeline.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

NUM_COURT_KEYPOINTS = 20


@dataclass
class CourtKPResult:
    """Result of court keypoint detection.

    Attributes:
        keypoints: Court keypoints (20, 2), normalized [0, 1].
        visibility: Keypoint visibility (20,).
        frame_index: Frame index used for detection.

    """

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    frame_index: int


class CourtKPModule(BasePipelineModule):
    """Court keypoint detection module.

    Detects 20 court keypoints from a single frame (fixed camera assumption).

    Attributes:
        checkpoint_path: Path to model checkpoint.
        device: Inference device.
        mode: "model" for predictor, "manual" for manual keypoints,
            "manual_ui" for interactive input.
        manual_keypoints_path: JSON file path for manual keypoints.

    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        mode: Literal["model", "manual", "manual_ui"] = "model",
        manual_keypoints_path: str | Path | None = None,
        manual_keypoints: NDArray[np.float32] | None = None,
        manual_visibility: NDArray[np.float32] | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the module.

        Args:
            checkpoint_path: Path to model checkpoint.
            mode: "model" to use predictor, "manual" for manual keypoints,
                "manual_ui" for interactive input.
            manual_keypoints_path: JSON file with manual keypoints.
            manual_keypoints: Manual keypoints array (20, 2).
            manual_visibility: Manual visibility array (20,).
            device: Inference device.

        """
        self.checkpoint_path = Path(checkpoint_path)
        self.mode = mode
        self.manual_keypoints_path = (
            Path(manual_keypoints_path) if manual_keypoints_path is not None else None
        )
        self.device = device
        self._predictor = None
        self._manual_keypoints: NDArray[np.float32] | None = None
        self._manual_visibility: NDArray[np.float32] | None = None
        self._manual_needs_normalization = False

        if manual_keypoints is not None:
            self._set_manual_keypoints(manual_keypoints, manual_visibility)

    def load(self) -> None:
        """Load the court keypoint predictor."""
        if self.mode in {"manual", "manual_ui"}:
            if self._manual_keypoints is not None:
                return
            if self.manual_keypoints_path is None:
                if self.mode == "manual":
                    raise ValueError(
                        "manual_keypoints_path must be set for manual mode when no "
                        "manual_keypoints are provided."
                    )
                return
            self._load_manual_keypoints(self.manual_keypoints_path)
            return

        if self._predictor is not None:
            return

        LOGGER.info(f"Loading Court KP model from {self.checkpoint_path}")
        from src.court_detection.inference.predictor import CourtKeypointPredictor

        self._predictor = CourtKeypointPredictor.from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        if self.mode in {"manual", "manual_ui"}:
            return self._manual_keypoints is not None
        return self._predictor is not None

    def _save_manual_keypoints(self, path: Path, image_size: tuple[int, int]) -> None:
        """Save manual keypoints to JSON in annotation format.

        Args:
            path: Output JSON path.
            image_size: Image size (width, height).
        """
        if self._manual_keypoints is None or self._manual_visibility is None:
            raise RuntimeError("Manual keypoints are not set.")

        width, height = image_size
        keypoints = []
        for i in range(NUM_COURT_KEYPOINTS):
            kp = self._manual_keypoints[i]
            vis = int(self._manual_visibility[i])
            if kp.max() <= 1.0:
                x = float(kp[0] * width)
                y = float(kp[1] * height)
            else:
                x = float(kp[0])
                y = float(kp[1])
            keypoints.append({"x": x, "y": y, "visibility": vis})

        data = {
            "image_width": width,
            "image_height": height,
            "num_keypoints": NUM_COURT_KEYPOINTS,
            "keypoints": keypoints,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _collect_manual_keypoints_ui(self, frame: NDArray[np.uint8]) -> None:
        """Collect manual keypoints via an interactive UI.

        Args:
            frame: RGB frame array (H, W, 3).
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required for manual_ui mode.") from exc

        keypoints = np.zeros((NUM_COURT_KEYPOINTS, 2), dtype=np.float32)
        visibility = np.zeros(NUM_COURT_KEYPOINTS, dtype=np.float32)
        current_idx = 0

        def draw_overlay(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
            overlay = image.copy()
            for idx in range(NUM_COURT_KEYPOINTS):
                if visibility[idx] <= 0:
                    continue
                x, y = keypoints[idx]
                color = (0, 255, 0) if visibility[idx] == 1 else (0, 165, 255)
                cv2.circle(overlay, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    str(idx),
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                overlay,
                f"Keypoint {current_idx}/19",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "LMB: visible | RMB: occluded | N/P: next/prev | S: save | Q: quit",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return overlay

        def on_mouse(event: int, mx: int, my: int, _flags: int, _param: object) -> None:
            nonlocal current_idx
            if event == cv2.EVENT_LBUTTONDOWN:
                keypoints[current_idx] = [mx, my]
                visibility[current_idx] = 1.0
                current_idx = (current_idx + 1) % NUM_COURT_KEYPOINTS
            elif event == cv2.EVENT_RBUTTONDOWN:
                keypoints[current_idx] = [mx, my]
                visibility[current_idx] = 2.0
                current_idx = (current_idx + 1) % NUM_COURT_KEYPOINTS

        window_name = "Court Keypoints (manual_ui)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                overlay = draw_overlay(frame)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(30) & 0xFF

                if key in {ord("n"), ord(" "), 82}:
                    current_idx = (current_idx + 1) % NUM_COURT_KEYPOINTS
                elif key in {ord("p"), 84}:
                    current_idx = (current_idx - 1) % NUM_COURT_KEYPOINTS
                elif key == ord("c"):
                    keypoints[current_idx] = [0.0, 0.0]
                    visibility[current_idx] = 0.0
                elif key == ord("s"):
                    break
                elif key in {ord("q"), 27}:
                    break
        finally:
            cv2.destroyWindow(window_name)

        self._manual_keypoints = keypoints.astype(np.float32)
        self._manual_visibility = visibility.astype(np.float32)
        self._manual_needs_normalization = True

    def _set_manual_keypoints(
        self,
        keypoints: NDArray[np.float32],
        visibility: NDArray[np.float32] | None = None,
    ) -> None:
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.shape != (NUM_COURT_KEYPOINTS, 2):
            raise ValueError(
                "manual_keypoints must have shape (20, 2), "
                f"got {keypoints.shape}."
            )

        if visibility is None:
            visibility = np.where(np.min(keypoints, axis=1) >= 0.0, 1.0, 0.0)
        else:
            visibility = np.asarray(visibility, dtype=np.float32)
            if visibility.shape != (NUM_COURT_KEYPOINTS,):
                raise ValueError(
                    "manual_visibility must have shape (20,), "
                    f"got {visibility.shape}."
                )

        self._manual_keypoints = keypoints.astype(np.float32)
        self._manual_visibility = visibility.astype(np.float32)
        self._manual_needs_normalization = bool(np.nanmax(self._manual_keypoints) > 1.0)

    def _load_manual_keypoints(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Manual keypoints file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        keypoints_raw = data.get("keypoints", [])
        keypoints = np.zeros((NUM_COURT_KEYPOINTS, 2), dtype=np.float32)
        visibility = np.zeros(NUM_COURT_KEYPOINTS, dtype=np.float32)

        if keypoints_raw and isinstance(keypoints_raw[0], dict):
            for i, kp in enumerate(keypoints_raw[:NUM_COURT_KEYPOINTS]):
                x = float(kp.get("x", 0.0))
                y = float(kp.get("y", 0.0))
                vis = float(kp.get("visibility", 0.0))
                if vis > 0:
                    keypoints[i] = [x, y]
                    visibility[i] = 1.0
        elif keypoints_raw:
            arr = np.asarray(keypoints_raw, dtype=np.float32)
            count = min(len(arr), NUM_COURT_KEYPOINTS)
            keypoints[:count] = arr[:count, :2]
            if "visibility" in data:
                visibility_raw = np.asarray(data.get("visibility"), dtype=np.float32)
                visibility[:count] = visibility_raw[:count]
            else:
                visibility[:count] = np.where(
                    np.min(keypoints[:count], axis=1) >= 0.0, 1.0, 0.0
                )

        self._manual_keypoints = keypoints.astype(np.float32)
        self._manual_visibility = visibility.astype(np.float32)

        image_width = data.get("image_width") or data.get("width")
        image_height = data.get("image_height") or data.get("height")
        if image_width and image_height:
            self._manual_keypoints[..., 0] /= float(image_width)
            self._manual_keypoints[..., 1] /= float(image_height)
            self._manual_needs_normalization = False
        else:
            self._manual_needs_normalization = bool(
                np.nanmax(self._manual_keypoints) > 1.0
            )

    def process(
        self,
        frame: NDArray[np.uint8],
        frame_index: int = 0,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> CourtKPResult:
        """Detect court keypoints from a frame.

        Args:
            frame: RGB frame array (H, W, 3).
            frame_index: Frame index (for metadata).
            image_width: Image width for normalization.
            image_height: Image height for normalization.

        Returns:
            CourtKPResult with normalized keypoints.

        """
        if self.mode in {"manual", "manual_ui"}:
            if not self.is_loaded:
                self.load()

            if self.mode == "manual_ui" and self._manual_keypoints is None:
                self._collect_manual_keypoints_ui(frame)
                if self.manual_keypoints_path is not None:
                    self._save_manual_keypoints(
                        self.manual_keypoints_path,
                        image_size=(frame.shape[1], frame.shape[0]),
                    )

            keypoints = np.array(self._manual_keypoints, copy=True)
            visibility = np.array(self._manual_visibility, copy=True)

            if self._manual_needs_normalization:
                if image_width is None or image_height is None:
                    raise ValueError(
                        "image_width and image_height are required to normalize "
                        "manual keypoints."
                    )
                keypoints[..., 0] /= image_width
                keypoints[..., 1] /= image_height

            return CourtKPResult(
                keypoints=keypoints.astype(np.float32),
                visibility=visibility.astype(np.float32),
                frame_index=frame_index,
            )

        if not self.is_loaded:
            self.load()

        result = self._predictor.predict(frame)

        keypoints = result["keypoints"].astype(np.float32)
        visibility = result["visibility"].astype(np.float32)

        if image_width is not None and image_height is not None:
            keypoints[..., 0] /= image_width
            keypoints[..., 1] /= image_height

        return CourtKPResult(
            keypoints=keypoints,
            visibility=visibility,
            frame_index=frame_index,
        )


if __name__ == "__main__":
    dummy_keypoints = np.zeros((NUM_COURT_KEYPOINTS, 2), dtype=np.float32)
    dummy_visibility = np.ones(NUM_COURT_KEYPOINTS, dtype=np.float32)
    dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    module = CourtKPModule(
        checkpoint_path="dummy.ckpt",
        mode="manual",
        manual_keypoints=dummy_keypoints,
        manual_visibility=dummy_visibility,
    )
    result = module.process(dummy_frame, image_width=64, image_height=64)
    assert result.keypoints.shape == (NUM_COURT_KEYPOINTS, 2)
    assert result.visibility.shape == (NUM_COURT_KEYPOINTS,)
    print("CourtKPModule smoke test passed.")
