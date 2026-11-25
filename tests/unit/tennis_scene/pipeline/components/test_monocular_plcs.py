"""Unit & integration tests for tennis_scene monocular PLCS (+ HTD-Refine) component."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.geometry.monocular_camera_bbox import (
    SyntheticBBoxConfig,
    compute_synthetic_bbx_xys,
    project_world_coco17_to_pixel,
)
from src.tasks.plcs.inference.htd_refiner import HTDRefinerConfig
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io.adapters import PLCSModelIOAdapter
from src.tasks.plcs.model_io.contracts import PLCSInputProfile
from src.tasks.plcs.models.plcs_monocular_gvhmr_model import PLCSMonocularGVHMRModel
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import (
    CalibrationSet,
    CameraGeometryConfig,
    LocalCourtCalibration,
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.monocular_plcs import (
    MonocularPLCSConfig,
    MonocularPLCSPlayerInput,
    MonocularPLCSReconstructionModule,
    SkeletonOnlyPlacementInput,
    SkeletonOnlyPlacementModule,
    build_single_view_geometry,
    reconstruct_single_view_monocular_scene,
)
from src.tennis_scene.pipeline.components.scene_assembly import (
    SceneAssemblyInput,
    SceneAssemblyModule,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.observation_types import (
    ObjectObservations,
    group_observations,
)
from src.tennis_scene.schema import validate_scene_result_arrays
from src.utils.geometry.keypoints import normalize_grid_keypoints
from src.utils.geometry.triangulation import PinholeCamera, PointRejection
from src.utils.projection.camera_projector import (
    MIXED_LAYOUT,
    CameraConfig,
    CameraProjector,
)
from src.utils.schema.court import (
    STANDARD_COURT_CONFIG,
    CourtConfig,
    court_keypoints_3d,
)


def _build_pinhole_cameras(count: int = 2) -> list[PinholeCamera]:
    proj = CameraProjector(
        config=CameraConfig(
            z_min=3.5,
            z_max=5.0,
            hfov_deg=60.0,
            image_size=(1280, 720),
            fixed_look_at=(0.0, 0.0, 0.0),
            fixed_baseline_clear_extra=0.0,
            fixed_position_noise_radius=0.0,
            fixed_look_at_xy_radius=0.0,
            layout=MIXED_LAYOUT,
            broadcast_setback=22.0,
            broadcast_height=8.0,
            broadcast_hfov_deg=35.0,
            broadcast_look_at_y=0.0,
            broadcast_look_at_height=0.5,
            broadcast_position_noise_radius=0.0,
            broadcast_look_at_xy_radius=0.0,
            broadcast_hfov_jitter_deg=0.0,
            broadcast_setback_range=(20.0, 26.0),
            broadcast_height_range=(7.0, 10.0),
            broadcast_court_width_frac_range=(0.55, 0.75),
        ),
        court_config=STANDARD_COURT_CONFIG,
    )
    cams: list[PinholeCamera] = []
    for idx, cam in enumerate(proj.cameras()[:count]):
        K = np.array(
            [
                [float(cam.f), 0.0, float(cam.cx)],
                [0.0, float(cam.f), float(cam.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        R = cam.R.numpy().astype(np.float64)
        C = cam.C.numpy().astype(np.float64)
        t = -(R @ C)
        cams.append(PinholeCamera(f"cam{idx}", K, R, t))
    return cams


def _synthetic_player_world_kp(seq_len: int = 16, fps: float = 30.0) -> torch.Tensor:
    t = torch.arange(seq_len, dtype=torch.float32) / fps
    root = torch.stack(
        [
            0.6 * torch.sin(2.0 * math.pi * 0.5 * t),
            -4.5 + 0.5 * t,
            torch.full_like(t, 0.92),
        ],
        dim=-1,
    )
    offsets = torch.tensor(
        [
            [0.00, -0.08, 0.66],
            [0.03, -0.10, 0.69],
            [-0.03, -0.10, 0.69],
            [0.07, -0.06, 0.67],
            [-0.07, -0.06, 0.67],
            [0.19, 0.00, 0.48],
            [-0.19, 0.00, 0.48],
            [0.28, -0.08, 0.24],
            [-0.28, -0.08, 0.24],
            [0.33, -0.20, 0.06],
            [-0.33, -0.20, 0.06],
            [0.12, 0.00, 0.00],
            [-0.12, 0.00, 0.00],
            [0.14, -0.06, -0.43],
            [-0.14, -0.06, -0.43],
            [0.14, 0.03, -0.88],
            [-0.14, 0.03, -0.88],
        ],
        dtype=torch.float32,
    )
    return root[:, None, :] + offsets[None, :, :]


def _build_tiny_predictor() -> PLCSPredictor:
    torch.manual_seed(13)
    model = PLCSMonocularGVHMRModel(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        max_seq_len=64,
        sliding_window_len=16,
        enable_pva_heads=True,
        predict_static_contact=True,
    )
    model.eval()
    adapter = PLCSModelIOAdapter(
        model_type=PLCSMonocularGVHMRModel,
        profile=PLCSInputProfile.MONOCULAR,
        num_court_tokens=14,
        camera_index=0,
        output_rank=3,
        predict_canonical_pose=True,
        predict_auxiliary_position=False,
        predict_world_pose=True,
        enable_pva_heads=True,
        predict_static_contact=True,
        max_sequence_length=64,
        court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
    )
    return PLCSPredictor(model=model, adapter=adapter, device=torch.device("cpu"))


def _make_clip_source(tmp_path: Path, camera_ids: tuple[str, ...], num_frames: int) -> ClipSource:
    videos = tuple(
        SourceVideo(
            camera_id=cid,
            path=tmp_path / f"{cid}.mp4",
            sha256="0" * 64,
            num_frames=num_frames,
            fps=30.0,
            width=1280,
            height=720,
        )
        for cid in camera_ids
    )
    return ClipSource(clip_id="clip_mono_test", videos=videos)


def _make_court_kp(cameras: list[PinholeCamera], num_frames: int) -> CourtKPResult:
    court_xyz = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14]
    kp_list = []
    vis_list = []
    for cam in cameras:
        uv_px, front = cam.project(court_xyz)
        kp_grid = normalize_grid_keypoints(uv_px.astype(np.float32), 1280, 720)
        kp_list.append(np.repeat(kp_grid[None, ...], num_frames, axis=0))
        vis_list.append(np.repeat(front.astype(np.float32)[None, ...], num_frames, axis=0))
    return CourtKPResult(
        keypoints=np.stack(kp_list, axis=0),
        visibility=np.stack(vis_list, axis=0),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        diagnostics={"output_keypoint_contract": "camera_view_v2"},
    )


def _make_observations_for_cameras(
    cameras: list[PinholeCamera],
    world_kp: torch.Tensor,
    observed_masks: list[np.ndarray],
    *,
    inject_gsi_boxes_on_unobserved: bool = False,
) -> ObjectObservations:
    views = len(cameras)
    frames = world_kp.shape[0]
    uv = np.zeros((views, frames, 1, 17, 2), dtype=np.float32)
    conf = np.zeros((views, frames, 1, 17), dtype=np.float32)
    obs = np.zeros((views, frames, 1), dtype=np.bool_)
    boxes: NDArray[np.float32] = np.zeros((views, frames, 1, 3), dtype=np.float32)
    ids: NDArray[np.int64] = np.zeros((views, 1), dtype=np.int64)

    for v, cam in enumerate(cameras):
        uv_t, in_front = project_world_coco17_to_pixel(
            world_kp.unsqueeze(0),
            R_w2c=torch.from_numpy(cam.rotation.astype(np.float32)).unsqueeze(0),
            C_w=torch.from_numpy(cam.center.astype(np.float32)).unsqueeze(0),
            fx=torch.tensor([float(cam.intrinsic[0, 0])]),
            fy=torch.tensor([float(cam.intrinsic[1, 1])]),
            cx0=torch.tensor([float(cam.intrinsic[0, 2])]),
            cy0=torch.tensor([float(cam.intrinsic[1, 2])]),
        )
        vis_all = in_front[0]
        bbx, _ = compute_synthetic_bbx_xys(
            uv_t[0],
            vis_all,
            config=SyntheticBBoxConfig(),
        )
        m = observed_masks[v]
        obs[v, :, 0] = m
        uv[v, :, 0] = uv_t[0].numpy()
        conf[v, :, 0] = np.where(m[:, None], 0.9, 0.0).astype(np.float32)
        if inject_gsi_boxes_on_unobserved:
            # Simulate GSI-interpolated non-zero boxes and 2D keypoints on unobserved frames!
            boxes[v, :, 0] = bbx.numpy()
            conf[v, ~m, 0, :] = 0.85
        else:
            boxes[v, m, 0] = bbx.numpy()[m]

    return ObjectObservations(
        camera_ids=tuple(c.camera_id for c in cameras),
        size=(1280, 720),
        fps=30.0,
        uv_px=uv,
        confidence=conf,
        observed=obs,
        local_track_ids=ids,
        boxes_xys=boxes,
    )


class TestMonocularPLCSPipelineComponent:
    def test_single_view_scene_v2_without_hmr2_or_smplx_and_excludes_gsi_frames(
        self, tmp_path: Path
    ) -> None:
        cams = _build_pinhole_cameras(count=1)
        seq_len = 16
        world_kp = _synthetic_player_world_kp(seq_len=seq_len, fps=30.0)
        # Frames 6..9 are unobserved (GSI-interpolated boxes present in raw.boxes_xys!).
        obs_mask: NDArray[np.bool_] = np.ones((seq_len,), dtype=np.bool_)
        obs_mask[6:10] = False

        raw_obs = _make_observations_for_cameras(
            cams, world_kp, [obs_mask], inject_gsi_boxes_on_unobserved=True
        )
        source = _make_clip_source(tmp_path, ("cam0",), seq_len)
        court = _make_court_kp(cams, seq_len)
        cal_set = CalibrationSet(
            views=(LocalCourtCalibration(cams[0], 0, 0, np.eye(3, dtype=np.float64), 1.5),),
            excluded={},
        )
        predictor = _build_tiny_predictor()
        cfg = MonocularPLCSConfig(
            htd_refine=True,
            htd_config=HTDRefinerConfig(num_steps=10),
        )

        # Verify monocular_plcs.py has zero HMR2 / GVHMR / SMPL-X dependencies in its source imports.
        src_text = (
            Path(__file__).resolve().parents[5]
            / "src/tennis_scene/pipeline/components/monocular_plcs.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("hmr2_features", "smpl_assets", "SmplVertexReconstructor", "SmplCoco17Reconstructor", "components.gvhmr"):
            assert forbidden not in src_text
        scene = reconstruct_single_view_monocular_scene(
            source=source,
            court=court,
            calibration=cal_set,
            camera_id="cam0",
            observations=raw_obs,
            predictor=predictor,
            config=cfg,
        )
        # Strictly passes SceneResult v2 schema contract!
        validate_scene_result_arrays(scene)
        assert scene.schema_version == 2
        assert scene.metadata["status"] == "ok"
        assert scene.player_smpl_valid is not None and not scene.player_smpl_valid.any()
        assert scene.smpl_vertices_local is None
        assert scene.player_observed is not None
        assert scene.player_valid is not None
        assert scene.player_kp_3d_vis is not None
        assert scene.player_kp_3d is not None

        # GSI-interpolated frames 6..9 MUST remain unobserved, invalid, and zeroed!
        assert not scene.player_observed[0, 6:10].any()
        assert not scene.player_valid[0, 6:10].any()
        assert not scene.player_kp_3d_vis[0, 6:10].any()
        assert np.all(scene.player_kp_3d[0, 6:10] == 0.0)
        assert np.all(scene.player_position[0, 6:10] == 0.0)

        # Observed frames 0..5 and 10..15 MUST be reconstructed and valid!
        assert scene.player_valid[0, :6].all() and scene.player_valid[0, 10:].all()
        assert scene.player_kp_3d_vis[0, :6].all() and scene.player_kp_3d_vis[0, 10:].all()

    def test_rejects_uncalibrated_camera_explicitly_with_zero_fallback(
        self, tmp_path: Path
    ) -> None:
        empty_cal = CalibrationSet(views=(), excluded={"cam0": "no_accepted_homography"})
        with pytest.raises(ReconstructionUnavailable, match="no accepted court calibration"):
            build_single_view_geometry(empty_cal, "cam0")

        cams = _build_pinhole_cameras(count=1)
        world_kp = _synthetic_player_world_kp(seq_len=8, fps=30.0)
        obs = _make_observations_for_cameras(cams, world_kp, [np.ones(8, dtype=np.bool_)])
        grouped = group_observations(obs, np.zeros((1, 8, 1), dtype=np.int64), threshold=0.15)
        source = _make_clip_source(tmp_path, ("cam0",), 8)
        module = MonocularPLCSReconstructionModule(
            ("cam0",),
            MonocularPLCSConfig(htd_refine=False),
            predictor=_build_tiny_predictor(),
        )
        assert module.model_specs() == {}
        with pytest.raises(ReconstructionUnavailable, match="calibrated aligned cameras"):
            module.process(
                MonocularPLCSPlayerInput(
                    source=source,
                    alignment=CameraAlignmentOutput(geometry=None),
                    raw=obs,
                    grouped=grouped,
                )
            )

    def test_multiview_component_and_skeleton_only_placement_assemble_scene_v2(
        self, tmp_path: Path
    ) -> None:
        cams = _build_pinhole_cameras(count=2)
        seq_len = 14
        world_kp = _synthetic_player_world_kp(seq_len=seq_len, fps=30.0)
        # View 0 misses frames 10..13 while view 1 still observes frames 10..13 (single-view occlusion recovery!).
        obs0: NDArray[np.bool_] = np.ones((seq_len,), dtype=np.bool_)
        obs0[10:14] = False
        obs1: NDArray[np.bool_] = np.ones((seq_len,), dtype=np.bool_)
        obs1[4:6] = False
        # Both views miss frame 7 -> must stay strictly invalid!
        obs0[7] = False
        obs1[7] = False

        raw_obs = _make_observations_for_cameras(
            cams, world_kp, [obs0, obs1], inject_gsi_boxes_on_unobserved=True
        )
        obj_ids = np.where(raw_obs.observed, 0, -1).astype(np.int64)
        grouped = group_observations(raw_obs, obj_ids, threshold=0.15)
        source = _make_clip_source(tmp_path, ("cam0", "cam1"), seq_len)
        court = _make_court_kp(cams, seq_len)

        cal_set = CalibrationSet(
            views=tuple(
                LocalCourtCalibration(c, idx, 0, np.eye(3, dtype=np.float64), 1.5)
                for idx, c in enumerate(cams)
            ),
            excluded={},
        )
        geom = resolve_camera_geometry(
            cal_set,
            "cam0",
            (False, False),
            (
                SideEvidence(
                    "plcs",
                    grouped.uv_px[:, :, :, [5, 6, 11, 12]],
                    grouped.visibility[:, :, :, [5, 6, 11, 12]],
                    40.0,
                ),
            ),
            config=CameraGeometryConfig(side_min_frames=2),
        )
        alignment = CameraAlignmentOutput(geometry=geom)
        cal_out = CourtCalibrationOutput(
            court=court,
            calibration=cal_set,
            reference_camera="cam0",
            footpoint_polygons={"cam0": None, "cam1": None},
        )

        mono_module = MonocularPLCSReconstructionModule(
            ("cam0", "cam1"),
            MonocularPLCSConfig(
                htd_refine=True,
                htd_config=HTDRefinerConfig(num_steps=10),
                use_multiview_reprojection_refine=True,
            ),
            predictor=_build_tiny_predictor(),
        )
        tri_out = mono_module.process(
            MonocularPLCSPlayerInput(
                source=source,
                alignment=alignment,
                raw=raw_obs,
                grouped=grouped,
            )
        )
        assert tri_out.skeleton is not None
        # Frame 7 (unobserved in all views) MUST be invalid with INSUFFICIENT_VIEWS!
        assert not tri_out.skeleton.valid[0, 7].any()
        assert (tri_out.skeleton.reasons[0, 7] == int(PointRejection.INSUFFICIENT_VIEWS)).all()
        # Frames 10..13 (observed ONLY in cam1, where DLT cannot triangulate!) ARE reconstructed!
        assert tri_out.skeleton.valid[0, 10:14].all()

        placement_module = SkeletonOnlyPlacementModule()
        placement_out = placement_module.process(SkeletonOnlyPlacementInput(skeleton=tri_out))
        assert placement_out.players is not None
        assert not placement_out.players.smpl_valid.any()
        assert placement_out.players.diagnostics["hmr2_used"] is False
        assert placement_out.players.diagnostics["smplx_used"] is False

        assembly = SceneAssemblyModule(
            ("cam0", "cam1"),
            require_people=True,
            require_ball=False,
            require_body=False,
            include_ball=False,
        )
        scene = assembly.process(
            SceneAssemblyInput(
                source=source,
                calibration=cal_out,
                alignment=alignment,
                people=grouped,
                skeleton=tri_out,
                ball=None,
                placement=placement_out,
            )
        )
        validate_scene_result_arrays(scene)
        assert scene.metadata["status"] == "ok"
        assert scene.player_valid is not None
        assert scene.player_valid[0, 10:14].all()
        assert not scene.player_valid[0, 7]
        assert scene.player_smpl_valid is not None and not scene.player_smpl_valid.any()

    def test_artifact_codec_roundtrip_and_model_specs_declare_only_plcs_checkpoint(
        self, tmp_path: Path
    ) -> None:
        from src.tennis_scene.pipeline.components.body_placement import (
            BodyPlacementOutput,
        )
        from src.tennis_scene.pipeline.components.triangulation import (
            PlayerTriangulationOutput,
        )
        from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
        from src.utils.configuration import PathResolver
        from src.utils.configuration.paths import RuntimePathRoots

        fake_ckpt = tmp_path / "monocular_plcs.ckpt"
        fake_ckpt.write_bytes(b"placeholder")
        fake_resolver = PathResolver(
            roots=RuntimePathRoots(
                project_root=tmp_path,
                data_root=tmp_path,
                checkpoint_root=tmp_path,
                output_root=tmp_path,
                artifact_root=tmp_path,
                cache_root=tmp_path,
                external_asset_root=tmp_path,
            )
        )
        mod = MonocularPLCSReconstructionModule(
            ("cam0",),
            MonocularPLCSConfig(checkpoint=fake_ckpt, htd_refine=False),
            path_resolver=fake_resolver,
        )
        specs = mod.model_specs()
        assert list(specs.keys()) == ["plcs_monocular"]
        assert specs["plcs_monocular"].weights == {"checkpoint": fake_ckpt}
        assert "hmr2" not in specs and "gvhmr" not in specs and "body" not in specs

        # Verify PlayerTriangulationOutput and BodyPlacementOutput round-trip via ArtifactCodec!
        cams = _build_pinhole_cameras(count=1)
        world_kp = _synthetic_player_world_kp(seq_len=6, fps=30.0)
        obs = _make_observations_for_cameras(cams, world_kp, [np.ones(6, dtype=np.bool_)])
        grouped = group_observations(obs, np.zeros((1, 6, 1), dtype=np.int64), threshold=0.15)
        source = _make_clip_source(tmp_path, ("cam0",), 6)
        cal_set = CalibrationSet(
            views=(LocalCourtCalibration(cams[0], 0, 0, np.eye(3, dtype=np.float64), 1.0),),
            excluded={},
        )
        geom, _ = build_single_view_geometry(cal_set, "cam0")
        direct_mod = MonocularPLCSReconstructionModule(
            ("cam0",),
            MonocularPLCSConfig(htd_refine=False),
            predictor=_build_tiny_predictor(),
        )
        tri_out = direct_mod.process(
            MonocularPLCSPlayerInput(
                source=source,
                alignment=CameraAlignmentOutput(geometry=geom),
                raw=obs,
                grouped=grouped,
            )
        )
        place_out = SkeletonOnlyPlacementModule().process(
            SkeletonOnlyPlacementInput(skeleton=tri_out)
        )

        tri_dir = tmp_path / "tri_artifact"
        tri_dir.mkdir()
        tri_codec = ArtifactCodec(PlayerTriangulationOutput)
        payload, arrays = tri_codec.dump(tri_out, tri_dir)
        restored_tri = tri_codec.load(payload, tri_dir, arrays)
        assert restored_tri.skeleton is not None and tri_out.skeleton is not None
        np.testing.assert_allclose(restored_tri.skeleton.positions, tri_out.skeleton.positions)

        place_dir = tmp_path / "place_artifact"
        place_dir.mkdir()
        place_codec = ArtifactCodec(BodyPlacementOutput)
        p_payload, p_arrays = place_codec.dump(place_out, place_dir)
        restored_place = place_codec.load(p_payload, place_dir, p_arrays)
        assert restored_place.players is not None and place_out.players is not None
        np.testing.assert_allclose(restored_place.players.position, place_out.players.position)
        assert restored_place.players.vertices_local is None
        assert not restored_place.players.smpl_valid.any()

    def test_model_spec_loader_loads_real_monocular_checkpoint(
        self, tmp_path: Path
    ) -> None:
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict

        from src.tasks.plcs.configuration import PLCSTrainingConfig
        from src.tasks.plcs.model_io import (
            build_plcs_model_io,
            write_plcs_checkpoint_court_keypoints,
        )
        from src.utils.configuration import PathResolver
        from src.utils.configuration.paths import RuntimePathRoots
        from src.utils.schema.court_normalization import (
            add_court_coordinate_normalization,
        )

        config_dir = Path("src/tasks/plcs/configs").resolve()
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "model=monocular_gvhmr_htd",
                    "data=monocular_sequence",
                    "loss=monocular_htd",
                ],
            )
        with open_dict(cfg):
            cfg.model.hidden_dim = 32
            cfg.model.num_layers = 1
            cfg.model.num_heads = 4
            cfg.model.mlp_ratio = 2.0

        runtime = PLCSTrainingConfig.from_config(cfg)
        bound = build_plcs_model_io(runtime)
        ckpt_path = tmp_path / "monocular_plcs_real.ckpt"
        payload: dict[str, object] = {
            "state_dict": {f"model.{k}": v for k, v in bound.model.state_dict().items()},
            "hyper_parameters": {"config": runtime.raw},
        }
        add_court_coordinate_normalization(payload, artifact="PLCS checkpoint")
        write_plcs_checkpoint_court_keypoints(
            payload, resolve_court_keypoint_contract("physical_v1")
        )
        torch.save(payload, ckpt_path)

        resolver = PathResolver(
            roots=RuntimePathRoots(
                project_root=tmp_path,
                data_root=tmp_path,
                checkpoint_root=tmp_path,
                output_root=tmp_path,
                artifact_root=tmp_path,
                cache_root=tmp_path,
                external_asset_root=tmp_path,
            )
        )
        mod = MonocularPLCSReconstructionModule(
            ("cam0",),
            MonocularPLCSConfig(checkpoint=ckpt_path, device="cpu", htd_refine=False),
            path_resolver=resolver,
        )
        loaded = mod.model_specs()["plcs_monocular"].factory()
        assert isinstance(loaded, PLCSPredictor)
        assert loaded.input_profile is PLCSInputProfile.MONOCULAR
