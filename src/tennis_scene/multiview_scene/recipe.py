"""An explicit downstream recipe over frozen 3D identities, cameras and ball."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.multiview_scene.inputs import INPUT_NODE, RECIPE, FrozenInputs
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.body_placement import (
    BodyPlacementInput,
    BodyPlacementModule,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionInput,
    BodyViewSelectionModule,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesModule
from src.tennis_scene.pipeline.components.scene_assembly import (
    SceneAssemblyInput,
    SceneAssemblyModule,
)
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationOutput,
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
)
from src.tennis_scene.pipeline.definition import declared_model_identities
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ComponentNode
from src.tennis_scene.pipeline.input_assembly.body import (
    GVHMRInputAssembler,
    HMR2FeaturesInputAssembler,
)
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.schema import SceneResult


class FrozenInput:
    io = ComponentIO(INPUT_NODE, type(None), FrozenInputs, {}, INPUT_NODE)

    def process(self, inputs: None) -> FrozenInputs:
        raise RuntimeError("The multiview recipe requires prepared inputs")


@dataclass(frozen=True)
class ViewInputs:
    version: int = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> BodyViewSelectionInput:
        inputs: FrozenInputs = artifacts["tracked"]
        return BodyViewSelectionInput(
            context.source, inputs.alignment, inputs.people.raw, inputs.people.grouped
        )


@dataclass(frozen=True)
class PlacementInputs:
    version: int = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> BodyPlacementInput:
        inputs: FrozenInputs = artifacts["tracked"]
        return BodyPlacementInput(
            context.source,
            inputs.alignment,
            inputs.people.raw,
            inputs.people.grouped,
            PlayerTriangulationOutput(inputs.people.skeleton),
            artifacts["recovered"],
        )


@dataclass(frozen=True)
class SceneInputs:
    version: int = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> SceneAssemblyInput:
        inputs: FrozenInputs = artifacts["tracked"]
        return SceneAssemblyInput(
            context.source,
            inputs.calibration,
            inputs.alignment,
            inputs.people.grouped,
            PlayerTriangulationOutput(inputs.people.skeleton),
            BallTriangulationOutput(inputs.ball),
            artifacts["placement"],
        )


class CheckedViews(BodyViewSelectionModule):
    def process(self, inputs: BodyViewSelectionInput) -> BodyViewSelectionOutput:
        result = super().process(inputs)
        missing = [
            person.person_id for person in result.selections if not person.requests
        ]
        if missing:
            raise ReconstructionUnavailable(
                "no_body_crops",
                f"No supported body crop series for selected IDs {missing}",
            )
        return result


class TrackedScene(SceneAssemblyModule):
    def __init__(
        self,
        source: ClipSource,
        reference: ArtifactRef,
        pins: dict[str, Any],
        sides: tuple[int, ...],
    ) -> None:
        super().__init__(
            source.camera_ids, require_people=True, require_ball=True, require_body=True
        )
        self.io = ComponentIO(
            "scene_assembly",
            SceneAssemblyInput,
            SceneResult,
            {
                "tracked": InputPort(INPUT_NODE),
                "placement": InputPort("placed_bodies"),
            },
            "scene_result",
            2,
        )
        self.metadata = {
            "player_reconstruction_method": RECIPE,
            "player_sides": list(sides),
            "players_per_side": pins["players_per_side"],
            "multiview_inputs": json_value(reference),
            "reused_ball": pins["ball"],
            "reused_alignment": pins["alignment"],
            "external_ball": pins["external_ball"],
        }

    def process(self, inputs: SceneAssemblyInput) -> SceneResult:
        result = super().process(inputs)
        result.metadata.update(self.metadata)
        placed = result.player_smpl_valid
        ball = result.ball_3d_valid
        if (
            placed is None
            or not placed.any(axis=1).all()
            or ball is None
            or not ball.any()
        ):
            result.metadata["status"] = "partial"
        return result


@dataclass(frozen=True)
class PreparedClip:
    source: ClipSource
    store: ClipStore
    ref: ArtifactRef
    pins: dict[str, Any]
    sides: tuple[int, ...]
    legacy_index: Path


def prepare_store(
    root: Path,
    source: ClipSource,
    inputs: FrozenInputs,
    legacy_index: Path,
    settings: dict[str, Any],
) -> PreparedClip:
    store = ClipStore(root, json_value(source), memory_entries=0)
    identity = {"recipe": RECIPE, "pins": inputs.pins, "adapter_settings": settings}
    ref = store.active(INPUT_NODE)
    if ref is None or ref.execution_key != document_digest(identity):
        ref = store.publish(
            INPUT_NODE,
            inputs,
            ArtifactCodec(FrozenInputs),
            schema=INPUT_NODE,
            version=1,
            identity=identity,
            dependencies={},
            provenance={"origin": "import", "recipe": RECIPE},
        )
    return PreparedClip(
        source, store, ref, inputs.pins, inputs.people.sides, legacy_index
    )


def make_clip_execution(
    clip: PreparedClip,
    cfg: PipelineRuntimeConfig,
    implementation: str,
    *,
    prepare_only: bool,
) -> ClipExecution:
    context = AssemblyContext(clip.source)
    track_input = {"tracked": INPUT_NODE}
    imported = FrozenInput()
    nodes = [
        # The runner binds the exact import before it can assemble any downstream input.
        ComponentNode(
            INPUT_NODE,
            imported,
            imported.io,
            ViewInputs(),
            {},
            context,
            {},
            implementation,
            "load",
            required_import_identity={
                "recipe": RECIPE,
                "pins": clip.pins,
                "adapter_settings": adapter_settings(cfg, implementation),
            },
        ),
    ]

    def add(
        component: Any,
        assembler: Any,
        bindings: dict[str, str],
        settings: Mapping[str, Any],
        *,
        after: tuple[str, ...] = (),
    ) -> None:
        stage = component.io.name
        nodes.append(
            ComponentNode(
                stage,
                component,
                component.io,
                assembler,
                bindings,
                context,
                json_value(
                    {
                        **settings,
                        "recipe": RECIPE,
                        "models": declared_model_identities(cfg, stage, component),
                    }
                ),
                implementation,
                after_components=after,
            )
        )

    views = CheckedViews(clip.source.camera_ids, max_frames=cfg.sampling_max_frames)
    views.io = replace(views.io, inputs={"tracked": InputPort(INPUT_NODE)})
    add(views, ViewInputs(), track_input, {"max_frames": cfg.sampling_max_frames})
    if not prepare_only:
        hmr2 = HMR2FeaturesModule(cfg.people)
        gvhmr = GVHMRModule(cfg.people)
        placement = BodyPlacementModule(
            clip.source.camera_ids,
            cfg.people,
            cfg.player_placement,
            reprojection_px=cfg.player_reprojection_px,
        )
        placement.io = replace(
            placement.io,
            inputs={
                "tracked": InputPort(INPUT_NODE),
                "recovered": InputPort("body_parameters", 2),
            },
        )
        add(
            hmr2, HMR2FeaturesInputAssembler(), {"selection": "body_view_selection"}, {}
        )
        add(
            gvhmr,
            GVHMRInputAssembler(),
            {"selection": "body_view_selection", "features": "hmr2_features"},
            {},
            after=("hmr2_features",),
        )
        add(
            placement,
            PlacementInputs(),
            {**track_input, "recovered": "gvhmr"},
            {
                "placement": cfg.player_placement,
                "reprojection_px": cfg.player_reprojection_px,
            },
            after=("gvhmr",),
        )
        add(
            TrackedScene(clip.source, clip.ref, clip.pins, clip.sides),
            SceneInputs(),
            {**track_input, "placement": "body_placement"},
            {},
        )
    return ClipExecution(clip.source.clip_id, nodes, clip.store)


def adapter_settings(cfg: PipelineRuntimeConfig, implementation: str) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "human_threshold": cfg.human_vis_threshold,
        "bbox_enlarge": cfg.people.runtime.tracking.bbox_enlarge,
    }
