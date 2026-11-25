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
    monocular_plcs_overrides,
    reconstruct_players_monocular_plcs,
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

    def test_rejects_non_canonical_intrinsic_matrices(self) -> None:
        """Finding #4: K with skew, K[2,2]!=1, or non-zero bottom-row off-diagonals must fail loudly."""
        base_cam = _build_pinhole_cameras(count=1)[0]
        world_kp = _synthetic_player_world_kp(seq_len=6, fps=30.0)
        obs = _make_observations_for_cameras(
            [base_cam], world_kp, [np.ones(6, dtype=np.bool_)]
        )
        grouped = group_observations(
            obs, np.zeros((1, 6, 1), dtype=np.int64), threshold=0.15
        )
        predictor = _build_tiny_predictor()
        cfg = MonocularPLCSConfig(htd_refine=False)

        bad_matrices = [
            np.array([[900.0, 12.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),  # skew
            np.array([[900.0, 0.0, 640.0], [5.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),   # K[1,0]!=0
            np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 2.0]]),   # K[2,2]!=1
            np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.01, 0.0, 1.0]]),  # K[2,0]!=0
        ]
        for bad_k in bad_matrices:
            bad_cam = PinholeCamera("cam0", bad_k, base_cam.rotation, base_cam.translation)
            with pytest.raises(ValueError, match="canonical zero-skew"):
                reconstruct_players_monocular_plcs(
                    grouped,
                    obs,
                    [bad_cam],
                    predictor=predictor,
                    fps=30.0,
                    config=cfg,
                )

    def test_invalid_view_prediction_is_excluded_from_stage_b_and_inliers(self) -> None:
        """Finding #2: invalid view prediction at t=3 must not affect Stage B or inliers."""
        from dataclasses import replace

        cams = _build_pinhole_cameras(count=2)
        seq_len = 8
        world_kp = _synthetic_player_world_kp(seq_len=seq_len, fps=30.0)
        base_pred = _build_tiny_predictor()

        class _InvalidFrameOnCam0Predictor:
            """Marks cam0 frame t=3 invalid (NaN + valid_frame_mask=False)."""

            def predict_monocular_sequence(self, **kwargs: object) -> object:
                res = base_pred.predict_monocular_sequence(**kwargs)  # type: ignore[arg-type]
                C_w = np.asarray(kwargs["C_w"], dtype=np.float64)
                if np.allclose(C_w, cams[0].center, atol=1e-3):
                    assert res.world_kp_3d_meters is not None
                    assert res.valid_frame_mask is not None
                    kp_mod = res.world_kp_3d_meters.copy()
                    mask_mod = res.valid_frame_mask.copy()
                    kp_mod[0, 3] = np.nan
                    mask_mod[0, 3] = False
                    return replace(
                        res,
                        world_kp_3d_meters=kp_mod,
                        valid_frame_mask=mask_mod,
                    )
                return res

        obs_clean = _make_observations_for_cameras(
            cams, world_kp, [np.ones(seq_len, dtype=np.bool_), np.ones(seq_len, dtype=np.bool_)]
        )
        uv_perturbed = obs_clean.uv_px.copy()
        uv_perturbed[0, 3, 0] += 300.0  # Huge 2D perturbation on cam0 frame 3!
        obs_perturbed = ObjectObservations(
            camera_ids=obs_clean.camera_ids,
            size=obs_clean.size,
            fps=obs_clean.fps,
            uv_px=uv_perturbed,
            confidence=obs_clean.confidence,
            observed=obs_clean.observed,
            local_track_ids=obs_clean.local_track_ids,
            boxes_xys=obs_clean.boxes_xys,
        )
        obj_ids: NDArray[np.int64] = np.zeros((2, seq_len, 1), dtype=np.int64)
        grp_clean = group_observations(obs_clean, obj_ids, threshold=0.15)
        grp_pert = group_observations(obs_perturbed, obj_ids, threshold=0.15)
        cfg = MonocularPLCSConfig(
            htd_refine=True,
            htd_config=HTDRefinerConfig(num_steps=12, reproj_weight=8.0),
            use_multiview_reprojection_refine=True,
        )
        wrapper = _InvalidFrameOnCam0Predictor()

        res_a = reconstruct_players_monocular_plcs(
            grp_clean, obs_clean, cams, predictor=wrapper, fps=30.0, config=cfg  # type: ignore[arg-type]
        )
        res_b = reconstruct_players_monocular_plcs(
            grp_pert, obs_perturbed, cams, predictor=wrapper, fps=30.0, config=cfg  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(
            res_a.skeleton.positions, res_b.skeleton.positions, atol=1e-5, rtol=1e-5
        )
        # cam0 (view 0) at frame 3 must NOT be an inlier and must have 0 reprojection diagnostic!
        assert not res_a.skeleton.inliers[0, 0, 3].any()
        assert not res_b.skeleton.inliers[0, 0, 3].any()
        assert np.all(res_a.skeleton.reprojection_px[0, 0, 3] == 0.0)
        assert np.array_equal(res_a.skeleton.inliers, res_b.skeleton.inliers)

    def test_view_switch_masks_unsupported_velocity_and_acceleration_in_stage_b(self) -> None:
        """Finding #3: view switch t=4->5 must not inject cam0 endpoint vel/acc or zero-vel bias."""
        from src.tasks.plcs.model_io.contracts import PLCSPhysicalPrediction

        cams = _build_pinhole_cameras(count=2)
        seq_len = 10
        fps = 30.0
        world_kp = _synthetic_player_world_kp(seq_len=seq_len, fps=fps).numpy()
        # cam0 observes frames 0..4, cam1 observes frames 5..9 (clean switch at 4->5!).
        obs0: NDArray[np.bool_] = np.zeros((seq_len,), dtype=np.bool_)
        obs0[:5] = True
        obs1: NDArray[np.bool_] = np.zeros((seq_len,), dtype=np.bool_)
        obs1[5:] = True

        raw_obs = _make_observations_for_cameras(
            cams, torch.from_numpy(world_kp), [obs0, obs1]
        )
        obj_ids = np.where(raw_obs.observed, 0, -1).astype(np.int64)
        grouped = group_observations(raw_obs, obj_ids, threshold=0.15)

        gt_vel = np.zeros_like(world_kp)
        gt_vel[:-1] = (world_kp[1:] - world_kp[:-1]) * fps
        gt_acc = np.zeros_like(world_kp)
        gt_acc[1:-1] = (world_kp[2:] - 2.0 * world_kp[1:-1] + world_kp[:-2]) * (fps**2)

        def _make_predictor(endpoint_spike: float) -> object:
            class _SwitchPredictor:
                def predict_monocular_sequence(self, **kwargs: object) -> PLCSPhysicalPrediction:
                    obs_in = np.asarray(kwargs["frame_observed"], dtype=np.bool_)
                    kp_out = np.where(obs_in[:, None, None], world_kp, np.nan)
                    vel_out = gt_vel.copy()
                    acc_out = gt_acc.copy()
                    C_w = np.asarray(kwargs["C_w"], dtype=np.float64)
                    if np.allclose(C_w, cams[0].center, atol=1e-3):
                        # Unsupervised endpoint of cam0 (t=4) for vel & acc!
                        vel_out[4] = endpoint_spike
                        acc_out[4] = endpoint_spike
                    else:
                        # Unsupervised startpoint of cam1 (t=5) for acc!
                        acc_out[5] = endpoint_spike
                    return PLCSPhysicalPrediction(
                        position_meters=kp_out[None, :, 11, :].astype(np.float32),
                        yaw_radians=np.zeros((1, seq_len), dtype=np.float32),
                        world_kp_3d_meters=kp_out[None].astype(np.float32),
                        pred_vel_3d_mps=vel_out[None].astype(np.float32),
                        pred_acc_3d_mps2=acc_out[None].astype(np.float32),
                        valid_frame_mask=obs_in[None],
                    )

            return _SwitchPredictor()

        cfg = MonocularPLCSConfig(
            htd_refine=True,
            htd_config=HTDRefinerConfig(
                num_steps=15,
                vel_weight=10.0,
                acc_weight=5.0,
                reproj_weight=0.0,
                bone_var_weight=0.0,
                bone_sym_weight=0.0,
                contact_weight=0.0,
                ground_penetration_weight=0.0,
                jerk_weight=0.0,
                init_anchor_weight=0.01,
            ),
        )
        res_clean = reconstruct_players_monocular_plcs(
            grouped, raw_obs, cams, predictor=_make_predictor(0.0), fps=fps, config=cfg  # type: ignore[arg-type]
        )
        res_spiked = reconstruct_players_monocular_plcs(
            grouped, raw_obs, cams, predictor=_make_predictor(1e6), fps=fps, config=cfg  # type: ignore[arg-type]
        )
        # Unsupervised endpoint spikes across the view switch must have ZERO effect!
        np.testing.assert_allclose(
            res_clean.skeleton.positions, res_spiked.skeleton.positions, atol=1e-5, rtol=1e-5
        )
        # And the view-switch step (t=4->5) must NOT be pulled toward zero velocity!
        np.testing.assert_allclose(
            res_clean.skeleton.positions[0], world_kp, atol=1e-3, rtol=1e-3
        )

    def test_relative_checkpoint_resolves_consistently_in_model_spec_and_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding #5: relative checkpoint resolves against checkpoint_root for both ModelSpec & factory."""
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict

        from src.tasks.plcs.configuration import PLCSTrainingConfig
        from src.tasks.plcs.model_io import (
            build_plcs_model_io,
            write_plcs_checkpoint_court_keypoints,
        )
        from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
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

        ckpt_root = (tmp_path / "checkpoints_root").resolve()
        real_ckpt = ckpt_root / "plcs_models" / "mono_rel.ckpt"
        real_ckpt.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "state_dict": {f"model.{k}": v for k, v in bound.model.state_dict().items()},
            "hyper_parameters": {"config": runtime.raw},
        }
        add_court_coordinate_normalization(payload, artifact="PLCS checkpoint")
        write_plcs_checkpoint_court_keypoints(
            payload, resolve_court_keypoint_contract("physical_v1")
        )
        torch.save(payload, real_ckpt)

        # Put a corrupted decoy file with the same relative path under a different CWD!
        cwd_dir = (tmp_path / "other_cwd").resolve()
        decoy = cwd_dir / "plcs_models" / "mono_rel.ckpt"
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_bytes(b"corrupted-decoy-not-a-checkpoint")
        monkeypatch.chdir(cwd_dir)

        resolver = PathResolver(
            roots=RuntimePathRoots(
                project_root=tmp_path.resolve(),
                data_root=tmp_path.resolve(),
                checkpoint_root=ckpt_root,
                output_root=tmp_path.resolve(),
                artifact_root=tmp_path.resolve(),
                cache_root=tmp_path.resolve(),
                external_asset_root=tmp_path.resolve(),
            )
        )
        mod = MonocularPLCSReconstructionModule(
            ("cam0",),
            MonocularPLCSConfig(
                checkpoint=Path("plcs_models/mono_rel.ckpt"),
                device="cpu",
                htd_refine=False,
            ),
            path_resolver=resolver,
        )
        specs = mod.model_specs()
        spec = specs["plcs_monocular"]
        assert spec.weights["checkpoint"] == real_ckpt
        # Verify InferenceRuntime hashes & loads the real checkpoint from checkpoint_root!
        node_key = ("clip", "player_triangulation")
        with InferenceRuntime({node_key: specs}) as model_rt, model_rt.node(
            node_key
        ) as borrower:
            loaded = borrower.borrow("plcs_monocular")
            assert isinstance(loaded, PLCSPredictor)
            assert loaded.input_profile is PLCSInputProfile.MONOCULAR

    def test_declarative_runner_executes_monocular_plcs_and_invalidates_stage_b_cache(
        self, tmp_path: Path
    ) -> None:
        """Findings #1 & #6: standard_definition + ComponentRunner end-to-end & Stage B cache invalidation."""
        from src.tennis_scene.pipeline.artifacts import json_value
        from src.tennis_scene.pipeline.components.camera_alignment import (
            CameraAlignmentModule,
        )
        from src.tennis_scene.pipeline.components.court_calibration import (
            CourtCalibrationModule,
        )
        from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
        from src.tennis_scene.pipeline.components.identity import (
            IDENTITIES_PORT,
            PlayerIdentitiesOutput,
        )
        from src.tennis_scene.pipeline.components.person_detection import (
            PersonDetectionModule,
            PersonDetectionOutput,
        )
        from src.tennis_scene.pipeline.components.person_features import (
            PersonAppearanceFeaturesModule,
            PersonAppearanceFeaturesOutput,
            PersonPoseFeaturesModule,
            PersonPoseFeaturesOutput,
        )
        from src.tennis_scene.pipeline.components.person_tracking import (
            PersonTrackingModule,
            PersonTrackingOutput,
        )
        from src.tennis_scene.pipeline.components.player_selection import (
            PlayerSelectionModule,
            PlayerSelectionOutput,
        )
        from src.tennis_scene.pipeline.components.pose_estimation import (
            PoseEstimationModule,
        )
        from src.tennis_scene.pipeline.contracts import ComponentIO
        from src.tennis_scene.pipeline.definition import (
            enabled_model_assets,
            standard_definition,
        )
        from src.tennis_scene.pipeline.runner import ComponentRunner
        from src.tennis_scene.pipeline.storage.clip_store import ClipStore
        from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
            FixedStage,
            runtime,
        )

        def _make_fixed(io: ComponentIO[object, object], result: object) -> FixedStage:
            stage = FixedStage(result)
            stage.io = io
            return stage

        def _upstream_overrides(
            cam_ids: tuple[str, ...],
            obs: ObjectObservations,
            court: CourtKPResult,
            cal_out: CourtCalibrationOutput,
            roi_margins: object,
        ) -> dict[str, object]:
            stages: dict[str, object] = {}
            frames = obs.num_frames
            count = obs.uv_px.shape[2]
            boxes = np.zeros((count, frames, 4), dtype=np.float32)
            boxes[..., 2:] = 100.0
            for v, cid in enumerate(cam_ids):
                local_court = CourtKPResult(
                    court.keypoints[v : v + 1, :1],
                    court.visibility[v : v + 1, :1],
                    np.array([0], dtype=np.int32),
                    {"output_keypoint_contract": "camera_view_v2"},
                )
                stages[f"court_detection/{cid}"] = _make_fixed(CourtKPModule.io, local_court)
                det = PersonDetectionOutput(
                    cid,
                    np.arange(frames + 1, dtype=np.int64) * count,
                    boxes.transpose(1, 0, 2).reshape(-1, 4),
                    np.ones(frames * count, dtype=np.float32),
                    np.arange(frames * count, dtype=np.int64),
                )
                stages[f"person_detection/{cid}"] = _make_fixed(PersonDetectionModule.io, det)
                stages[f"person_pose_features/{cid}"] = _make_fixed(
                    PersonPoseFeaturesModule.io,
                    PersonPoseFeaturesOutput(
                        cid,
                        det.frame_offsets,
                        det.source_rows,
                        np.zeros((frames * count, 17, 3), dtype=np.float32),
                        "a" * 64,
                        "b" * 64,
                    ),
                )
                stages[f"person_appearance_features/{cid}"] = _make_fixed(
                    PersonAppearanceFeaturesModule.io,
                    PersonAppearanceFeaturesOutput(
                        cid,
                        det.frame_offsets,
                        det.source_rows,
                        np.zeros((frames * count, 1280), dtype=np.float32),
                        np.zeros(frames * count, dtype=np.bool_),
                        "clipreid_vitb16_market1501",
                        "a" * 64,
                        "b" * 64,
                        np.zeros((frames, 32), dtype=np.uint8),
                    ),
                )
                tracks = PersonTrackingOutput(
                    cid,
                    np.arange(count, dtype=np.int64),
                    boxes,
                    obs.observed[v].T,
                    tuple((i,) for i in range(count)),
                    (),
                )
                stages[f"person_tracking/{cid}"] = _make_fixed(PersonTrackingModule.io, tracks)
                origins = np.broadcast_to(
                    np.arange(count, dtype=np.int64)[:, None], (count, frames)
                ).copy()
                origins[~tracks.observed] = -1
                stages[f"player_selection/{cid}"] = _make_fixed(
                    PlayerSelectionModule.io,
                    PlayerSelectionOutput(cid, tracks.track_ids, tracks.observed, tracks, origins, {}),
                )
                stages[f"pose_estimation/{cid}"] = _make_fixed(
                    PoseEstimationModule.io, obs.select_views((v,))
                )
            cal_mod = CourtCalibrationModule(
                cam_ids, CameraGeometryConfig(), roi_margins=roi_margins  # type: ignore[arg-type]
            )
            stages["court_calibration"] = _make_fixed(cal_mod.io, cal_out)
            return stages

        # --- Part A: Single-view declarative runner -> SceneResult v2 (ZERO HMR2/SMPL-X assets!) ---
        cams_1 = _build_pinhole_cameras(count=1)
        seq_len = 10
        world_kp = _synthetic_player_world_kp(seq_len=seq_len, fps=30.0)
        obs_1 = _make_observations_for_cameras(
            cams_1, world_kp, [np.ones(seq_len, dtype=np.bool_)]
        )
        court_1 = _make_court_kp(cams_1, seq_len)
        cal_set_1 = CalibrationSet(
            views=(LocalCourtCalibration(cams_1[0], 0, 0, np.eye(3, dtype=np.float64), 1.2),),
            excluded={},
        )
        cal_out_1 = CourtCalibrationOutput(
            court=court_1,
            calibration=cal_set_1,
            reference_camera="cam0",
            footpoint_polygons={"cam0": None},
        )

        sv_dir = tmp_path / "sv"
        sv_dir.mkdir()
        cfg = runtime(
            sv_dir,
            overrides=(
                "ball_detection.enabled=false",
                "ball_reconstruction.enabled=false",
                "court_side.evidence=player",
                "gvhmr.enabled=false",
            ),
        )
        assert not any(
            k in enabled_model_assets(cfg) for k in ("hmr2", "gvhmr", "smpl", "smplx")
        )
        source_1 = _make_clip_source(sv_dir, ("cam0",), seq_len)
        store_1 = ClipStore(sv_dir / "store", json_value(source_1))
        predictor = _build_tiny_predictor()
        cfg_a = MonocularPLCSConfig(
            htd_refine=True,
            htd_config=HTDRefinerConfig(num_steps=6, conf_threshold=0.2),
        )
        ov_1 = {
            **_upstream_overrides(("cam0",), obs_1, court_1, cal_out_1, cfg.person_roi_margins),
            **monocular_plcs_overrides(
                ("cam0",),
                cfg_a,
                predictor=predictor,
                single_view_camera_id="cam0",
            ),
        }
        nodes_1 = standard_definition(cfg, source_1, code_identity="test", overrides=ov_1)
        runner_1 = ComponentRunner(nodes_1, store_1)
        runner_1.run(targets=("scene_assembly",))
        scene_1 = runner_1.output("scene_assembly")
        validate_scene_result_arrays(scene_1)
        assert scene_1.schema_version == 2
        assert scene_1.metadata["status"] == "ok"
        assert scene_1.player_valid is not None and scene_1.player_valid.all()
        assert scene_1.player_smpl_valid is not None and not scene_1.player_smpl_valid.any()
        assert scene_1.smpl_vertices_local is None

        # --- Part B (Finding #6): Changing conf_threshold / min_joint_z_m / jerk_weight MUST invalidate cache! ---
        cfg_b = MonocularPLCSConfig(
            htd_refine=True,
            htd_config=HTDRefinerConfig(num_steps=6, conf_threshold=0.5),
        )
        assert cfg_a.cache_identity() != cfg_b.cache_identity()
        assert (
            MonocularPLCSConfig(htd_config=HTDRefinerConfig(min_joint_z_m=0.0)).cache_identity()
            != MonocularPLCSConfig(htd_config=HTDRefinerConfig(min_joint_z_m=0.15)).cache_identity()
        )
        assert (
            MonocularPLCSConfig(htd_config=HTDRefinerConfig(jerk_weight=0.05)).cache_identity()
            != MonocularPLCSConfig(htd_config=HTDRefinerConfig(jerk_weight=0.25)).cache_identity()
        )

        runner_same = ComponentRunner(nodes_1, store_1)
        runner_same.run(targets=("scene_assembly",))
        assert runner_same.statuses["player_triangulation"] == "cached"
        assert runner_same.statuses["scene_assembly"] == "cached"

        ov_1_b = {
            **ov_1,
            **monocular_plcs_overrides(
                ("cam0",),
                cfg_b,
                predictor=predictor,
                single_view_camera_id="cam0",
            ),
        }
        nodes_1_b = standard_definition(cfg, source_1, code_identity="test", overrides=ov_1_b)
        runner_diff = ComponentRunner(nodes_1_b, store_1)
        runner_diff.run(targets=("scene_assembly",))
        assert runner_diff.statuses["player_triangulation"] == "executed"
        assert runner_diff.statuses["scene_assembly"] == "executed"

        # --- Part C: Multi-view declarative runner -> SceneResult v2 via standard_definition ---
        cams_2 = _build_pinhole_cameras(count=2)
        obs_2 = _make_observations_for_cameras(
            cams_2, world_kp, [np.ones(seq_len, dtype=np.bool_), np.ones(seq_len, dtype=np.bool_)]
        )
        court_2 = _make_court_kp(cams_2, seq_len)
        cal_set_2 = CalibrationSet(
            views=tuple(
                LocalCourtCalibration(c, idx, 0, np.eye(3, dtype=np.float64), 1.2)
                for idx, c in enumerate(cams_2)
            ),
            excluded={},
        )
        cal_out_2 = CourtCalibrationOutput(
            court=court_2,
            calibration=cal_set_2,
            reference_camera="cam0",
            footpoint_polygons={"cam0": None, "cam1": None},
        )
        obj_ids_2: NDArray[np.int64] = np.zeros((2, seq_len, 1), dtype=np.int64)
        grouped_2 = group_observations(obs_2, obj_ids_2, threshold=0.15)
        geom_2 = resolve_camera_geometry(
            cal_set_2,
            "cam0",
            (False, False),
            (
                SideEvidence(
                    "plcs",
                    grouped_2.uv_px[:, :, :, [5, 6, 11, 12]],
                    grouped_2.visibility[:, :, :, [5, 6, 11, 12]],
                    40.0,
                ),
            ),
            config=CameraGeometryConfig(side_min_frames=2),
        )
        mv_dir = tmp_path / "mv"
        mv_dir.mkdir()
        cfg_mv = runtime(
            mv_dir,
            overrides=(
                "ball_detection.enabled=false",
                "ball_reconstruction.enabled=false",
                "court_side.evidence=player",
                "gvhmr.enabled=false",
            ),
        )
        source_2 = _make_clip_source(mv_dir, ("cam0", "cam1"), seq_len)
        store_2 = ClipStore(mv_dir / "store", json_value(source_2))

        from src.tasks.court_side.hypothesis import HypothesisScore
        from src.tennis_scene.pipeline.components.court_side import (
            SIDE_PORT,
            CourtSideOutput,
            PlayerCourtSideModule,
        )
        from src.tennis_scene.pipeline.contracts import InputPort

        side_mod = PlayerCourtSideModule(
            ("cam0", "cam1"), cfg_mv.court_side, max_frames=cfg_mv.sampling_max_frames
        )
        side_out = CourtSideOutput(
            camera_ids=("cam0", "cam1"),
            reference_camera="cam0",
            view_half_turns=(False, False),
            hypotheses=(
                HypothesisScore((False, False), cost=0.05, support=1.0, frames=seq_len),
                HypothesisScore((False, True), cost=0.95, support=0.1, frames=seq_len),
            ),
            margin=0.9,
            frames=seq_len,
        )
        ids_io = ComponentIO(
            "player_association",
            object,
            PlayerIdentitiesOutput,
            {
                "calibration": InputPort("local_court_calibration"),
                "side": SIDE_PORT,
                "tracks_cam0": InputPort("selected_player_tracks", 3),
                "tracks_cam1": InputPort("selected_player_tracks", 3),
            },
            IDENTITIES_PORT.schema,
            IDENTITIES_PORT.version,
        )
        ids_out_2 = PlayerIdentitiesOutput(
            camera_ids=("cam0", "cam1"),
            local_track_ids=obs_2.local_track_ids,
            player_ids=np.zeros((2, 1, seq_len), dtype=np.int64),
            diagnostics={"method": "test_fixed"},
        )
        align_mod = CameraAlignmentModule(
            ("cam0", "cam1"),
            cfg_mv.camera_geometry,
            player_reprojection_px=cfg_mv.player_reprojection_px,
            ball_reprojection_px=cfg_mv.ball_reprojection_px,
            joint_confidence=cfg_mv.joint_confidence,
            max_frames=cfg_mv.sampling_max_frames,
            include_ball=False,
        )
        ov_2: dict[str, object] = {
            **_upstream_overrides(("cam0", "cam1"), obs_2, court_2, cal_out_2, cfg_mv.person_roi_margins),
            "court_side": _make_fixed(side_mod.io, side_out),
            "player_association": _make_fixed(ids_io, ids_out_2),
            "camera_alignment": _make_fixed(
                align_mod.io, CameraAlignmentOutput(geometry=geom_2)
            ),
            **monocular_plcs_overrides(
                ("cam0", "cam1"),
                cfg_a,
                predictor=predictor,
            ),
        }
        nodes_2 = standard_definition(cfg_mv, source_2, code_identity="test", overrides=ov_2)
        runner_2 = ComponentRunner(nodes_2, store_2)
        runner_2.run(targets=("scene_assembly",))
        scene_2 = runner_2.output("scene_assembly")
        validate_scene_result_arrays(scene_2)
        assert scene_2.schema_version == 2
        assert scene_2.metadata["status"] == "ok"
        assert scene_2.player_valid is not None and scene_2.player_valid.all()
        assert scene_2.player_smpl_valid is not None and not scene_2.player_smpl_valid.any()
