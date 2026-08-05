"""Helpers for assembling BLCS generator configuration objects."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.tasks.blcs.generate_dataset.simulation.ball_physics import PhysicsConfig
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyConfig
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
)
from src.utils.projection.camera_projector import CameraConfig
from src.utils.schema.court import CourtConfig


def build_generator_config(cfg: DictConfig) -> GeneratorConfig:
    """Build a BLCS generator config from a composed OmegaConf config."""
    physics_config = PhysicsConfig(
        gravity=float(cfg.physics.gravity),
        k_drag=float(cfg.physics.k_drag),
        k_magnus=float(cfg.physics.k_magnus),
        e_z=float(cfg.physics.e_z),
        mu=float(cfg.physics.mu),
        alpha_net=float(cfg.physics.alpha_net),
        alpha_net_cord=float(cfg.physics.alpha_net_cord),
        alpha_fence=float(cfg.physics.alpha_fence),
        net_half_thickness=float(cfg.physics.net_half_thickness),
        net_cord_radius=float(cfg.physics.net_cord_radius),
        dt=float(cfg.physics.dt),
        use_drag=bool(cfg.physics.use_drag),
        use_magnus=bool(cfg.physics.use_magnus),
        wind=tuple(cfg.physics.wind),
        gravity_range=tuple(cfg.physics.gravity_range),
        k_drag_range=tuple(cfg.physics.k_drag_range),
        k_magnus_range=tuple(cfg.physics.k_magnus_range),
        e_z_range=tuple(cfg.physics.e_z_range),
        mu_range=tuple(cfg.physics.mu_range),
        wind_speed_range=tuple(cfg.physics.wind_speed_range),
        wind_direction_range_deg=tuple(cfg.physics.wind_direction_range_deg),
    )

    rally_config = RallyConfig(
        z_range=tuple(cfg.rally.z_range),
        spin_x_range=tuple(cfg.rally.spin_x_range),
        spin_y_range=tuple(cfg.rally.spin_y_range),
        spin_z_range=tuple(cfg.rally.spin_z_range),
        max_sim_frames=int(cfg.rally.max_sim_frames),
        output_fps=int(cfg.rally.output_fps),
        sim_fps=int(cfg.rally.sim_fps),
        max_rallies=int(cfg.rally.max_rallies),
        max_total_frames=int(cfg.rally.max_total_frames),
        hit_timing_range=tuple(cfg.rally.hit_timing_range),
        return_z_range=tuple(cfg.rally.return_z_range),
        serve_probability=float(cfg.rally.serve_probability),
        serve_z_range=tuple(cfg.rally.serve_z_range),
        toss_vz_range=tuple(cfg.rally.toss_vz_range),
        toss_xy_noise_range=tuple(cfg.rally.toss_xy_noise_range),
        toss_max_frames=int(cfg.rally.toss_max_frames),
        toss_z0_tolerance=float(cfg.rally.toss_z0_tolerance),
        volley_probability=float(cfg.rally.volley_probability),
        normal_return_probability=float(cfg.rally.normal_return_probability),
        late_return_probability=float(cfg.rally.late_return_probability),
        out_court_target_probability=float(cfg.rally.out_court_target_probability),
    )

    camera_config = CameraConfig(
        z_min=float(cfg.camera.z_min),
        z_max=float(cfg.camera.z_max),
        hfov_deg=float(cfg.camera.hfov_deg),
        image_size=tuple(cfg.camera.image_size),
        fixed_look_at=tuple(cfg.camera.fixed_look_at),
        fixed_baseline_clear_extra=float(cfg.camera.fixed_baseline_clear_extra),
        fixed_position_noise_radius=float(cfg.camera.fixed_position_noise_radius),
        fixed_look_at_xy_radius=float(cfg.camera.fixed_look_at_xy_radius),
        layout=str(cfg.camera.layout),
        broadcast_setback=float(cfg.camera.broadcast_setback),
        broadcast_height=float(cfg.camera.broadcast_height),
        broadcast_hfov_deg=float(cfg.camera.broadcast_hfov_deg),
        broadcast_look_at_y=float(cfg.camera.broadcast_look_at_y),
        broadcast_look_at_height=float(cfg.camera.broadcast_look_at_height),
        broadcast_position_noise_radius=float(
            cfg.camera.broadcast_position_noise_radius
        ),
        broadcast_look_at_xy_radius=float(cfg.camera.broadcast_look_at_xy_radius),
        broadcast_hfov_jitter_deg=float(cfg.camera.broadcast_hfov_jitter_deg),
        broadcast_setback_range=(
            tuple(cfg.camera.broadcast_setback_range)
            if cfg.camera.broadcast_setback_range is not None
            else None
        ),
        broadcast_height_range=(
            tuple(cfg.camera.broadcast_height_range)
            if cfg.camera.broadcast_height_range is not None
            else None
        ),
        broadcast_court_width_frac_range=(
            tuple(cfg.camera.broadcast_court_width_frac_range)
            if cfg.camera.broadcast_court_width_frac_range is not None
            else None
        ),
    )

    court_cfg = cfg.generator.court
    court_config = CourtConfig(
        net_post_offset_x=float(court_cfg.net_post_offset_x),
        net_post_offset_x_range=tuple(court_cfg.net_post_offset_x_range),
    )

    targeted_velocity_config = TargetedVelocityConfig(
        drive_elevation_range_deg=tuple(
            cfg.targeted_velocity.drive_elevation_range_deg
        ),
        lob_elevation_range_deg=tuple(cfg.targeted_velocity.lob_elevation_range_deg),
        lob_probability=float(cfg.targeted_velocity.lob_probability),
        max_ballistic_apex_height_m=float(
            cfg.targeted_velocity.max_ballistic_apex_height_m
        ),
        gravity=float(cfg.targeted_velocity.gravity),
        net_retry_max_attempts=int(cfg.targeted_velocity.net_retry_max_attempts),
        net_check_max_frames=int(cfg.targeted_velocity.net_check_max_frames),
        net_elevation_step_deg=float(cfg.targeted_velocity.net_elevation_step_deg),
        landing_refine_enabled=bool(cfg.targeted_velocity.landing_refine_enabled),
        landing_refine_max_iters=int(cfg.targeted_velocity.landing_refine_max_iters),
        landing_refine_tolerance_m=float(
            cfg.targeted_velocity.landing_refine_tolerance_m
        ),
        landing_sim_max_frames=int(cfg.targeted_velocity.landing_sim_max_frames),
        target_margin_m=float(cfg.targeted_velocity.target_margin_m),
    )

    return GeneratorConfig(
        physics=physics_config,
        rally=rally_config,
        camera=camera_config,
        targeted_velocity=targeted_velocity_config,
        court=court_config,
    )
