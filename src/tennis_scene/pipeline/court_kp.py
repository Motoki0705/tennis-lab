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
class CourtKPConfig:
    """Configuration for CourtKP module.

    Attributes:
        checkpoint_path: Path to model checkpoint.
        mode: Detection mode ("model", "manual_ui").
        device: Inference device.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint_path: str | Path
    mode: Literal["model", "manual_ui"] = "model"
    device: str = "cuda"
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


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

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "keypoints": self.keypoints.tolist(),
            "visibility": self.visibility.tolist(),
            "frame_index": self.frame_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CourtKPResult":
        """Create result from dict."""
        return cls(
            keypoints=np.array(data["keypoints"], dtype=np.float32),
            visibility=np.array(data["visibility"], dtype=np.float32),
            frame_index=data["frame_index"],
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved CourtKP result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CourtKPResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class CourtKPModule(BasePipelineModule):
    """Court keypoint detection module.

    Detects 20 court keypoints from a single frame (fixed camera assumption).

    Attributes:
        config: Configuration for the module.
        checkpoint_path: Path to model checkpoint.
        device: Inference device.
        mode: "model" for predictor or "manual_ui" for interactive input.

    """

    def __init__(
        self,
        config: CourtKPConfig | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        mode: Literal["model", "manual_ui"] = "model",
        device: str = "cuda",
        save_result: bool = False,
        output_path: str | Path | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            config: CourtKP configuration (preferred).
            checkpoint_path: Path to model checkpoint (legacy).
            mode: "model" to use predictor or "manual_ui" for interactive input.
            device: Inference device (legacy).
            save_result: Whether to save result (legacy).
            output_path: Path to save result (legacy).

        """
        if config is not None:
            self.config = config
        else:
            if checkpoint_path is None:
                raise ValueError("Either config or checkpoint_path must be provided")
            self.config = CourtKPConfig(
                checkpoint_path=checkpoint_path,
                mode=mode,
                device=device,
                save_result=save_result,
                output_path=output_path,
            )
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.mode = self.config.mode
        self.device = self.config.device
        self._predictor = None
        self._manual_keypoints: NDArray[np.float32] | None = None
        self._manual_visibility: NDArray[np.float32] | None = None
        self._manual_needs_normalization = False

    def load(self) -> None:
        """Load the court keypoint predictor."""
        if self.mode == "manual_ui":
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
        if self.mode == "manual_ui":
            return self._manual_keypoints is not None
        return self._predictor is not None

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
        # Check if we should load from pre-computed result
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading CourtKP result from {load_path} (skipping detection)")
                return CourtKPResult.load(load_path)
            else:
                LOGGER.warning(f"load_path specified but not found: {load_path}, running detection")

        if self.mode == "manual_ui":
            if not self.is_loaded:
                self.load()

            if self._manual_keypoints is None:
                self._collect_manual_keypoints_ui(frame)

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

            result = CourtKPResult(
                keypoints=keypoints.astype(np.float32),
                visibility=visibility.astype(np.float32),
                frame_index=frame_index,
            )

            if self.config.save_result and self.config.output_path is not None:
                result.save(self.config.output_path)

            return result

        if not self.is_loaded:
            self.load()

        result = self._predictor.predict(frame)

        keypoints = result["keypoints"].astype(np.float32)
        visibility = result["visibility"].astype(np.float32)

        if image_width is not None and image_height is not None:
            keypoints[..., 0] /= image_width
            keypoints[..., 1] /= image_height

        result = CourtKPResult(
            keypoints=keypoints,
            visibility=visibility,
            frame_index=frame_index,
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result


if __name__ == "__main__":
    # Quick smoke test for module instantiation
    print("CourtKPModule: court keypoint detection module")
    print("Use CourtKPModule(CourtKPConfig(...)) to create")

    # Test config creation
    config = CourtKPConfig(
        checkpoint_path="test.ckpt",
        mode="manual_ui",
        device="cpu",
        save_result=True,
        output_path="test_output.json",
    )
    print(f"Config: {config}")
    assert config.device == "cpu"
    assert config.mode == "manual_ui"
    print("Smoke test passed.")
