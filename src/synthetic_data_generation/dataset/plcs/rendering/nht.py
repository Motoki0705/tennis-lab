"""PLCS adapter over the public ``NHTRenderClient`` file boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.dataset.plcs.rendering.contracts import (
    PLCSForegroundCompositor,
)
from src.synthetic_data_generation.dataset.runtime import (
    RenderSession,
    SharedBackgroundStore,
)
from src.synthetic_data_generation.rendering.nht.client import NHTRenderClient
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True, slots=True)
class NHTPLCSRenderer:
    """Render canonical backgrounds publicly, then require composed Gaussian AOVs."""

    client: NHTRenderClient
    compositor: PLCSForegroundCompositor
    executable: str | Path
    environment: Mapping[str, str]
    timeout_seconds: float

    def render_background_store(
        self,
        *,
        scene_path: Path,
        cameras: tuple[SceneCamera, ...],
        metric_adapter: MetricSceneAdapter,
        staging_directory: Path,
        session: RenderSession,
    ) -> SharedBackgroundStore:
        """Invoke public NHT exactly once and publish one shared rig background."""
        if not cameras:
            raise ValueError("PLCS rendering requires at least one generated camera.")
        if session.domain != "plcs" or session.nht_invocations != 0:
            raise ValueError("PLCS requires a fresh single-invocation render session.")
        output = staging_directory / "nht-backgrounds"
        request_path = staging_directory / "nht-background-cameras.json"
        if not isinstance(metric_adapter, MetricSceneAdapter):
            raise TypeError("PLCS rendering requires the alignment metric adapter.")
        arbitrary = NHTRenderRequest(
            cameras=tuple(
                NHTRenderCamera(
                    camera_id=camera.camera_id,
                    width=camera.width,
                    height=camera.height,
                    intrinsics=camera.intrinsics,
                    camera_to_scene=metric_adapter.nht_from_metric_camera(
                        camera.camera_to_scene
                    ),
                )
                for camera in cameras
            )
        )
        command = NHTRenderCommandRequest(
            scene_path=scene_path,
            output_directory=output,
            arbitrary_cameras=arbitrary,
            arbitrary_request_path=request_path,
            executable=self.executable,
        )
        session.note_nht_invocation()
        rendered = self.client.render(
            command,
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        return session.create_background_store(
            "rig",
            staging_directory / "backgrounds",
            rendered=rendered,
            nht_scene_units_per_metre=metric_adapter.nht_scene_units_per_metre,
            expected_camera_ids=tuple(camera.camera_id for camera in cameras),
        )


__all__ = ["NHTPLCSRenderer"]
