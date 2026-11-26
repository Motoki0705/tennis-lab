"""Replay the synthetic ball-event threshold calibration and held-out test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.data.augmentation.occlusion import sample_occlusion
from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallState
from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallySimulator
from src.tennis_scene.ball_events import RuleThresholds, detect_ball_events
from src.tennis_scene.ball_events.rules import (
    DEFAULT_THRESHOLDS,
    EVENT_KINDS,
    EventKind,
)
from src.utils.io import write_json_atomic
from src.utils.physics.ball.events import event_detection_report
from src.utils.physics.ball.record import EVENT_BITS
from src.utils.schema.court import net_height_at_x

MANIFEST_SHA256 = "a7880d12ac2475f31b607d3c654209f9c7f3e43342db61eb99244cce7ccdf951"
CONFIG_SHA256 = "943f4e5b37b10d84b30bfd6208fb9effdaf61049ab673610818b8f731da4b150"
SIM_SEED = {"val": 1207, "test": 5309}
MASK_SEED = {"val": 2718, "test": 3141}
TOLERANCE_FRAMES = 3
NET_CASES_PER_GROUP = 32
JITTER_M = 0.015
CORRUPTION = CorruptionConfig(0.5, 0.01, 3, 10, 0.0, 0.0, 0.0, 3)

# Independent starting points for the validation sweep, including the values
# later selected as code defaults. The cutoff for player proximity is geometric.
INITIAL_THRESHOLDS = RuleThresholds(
    velocity_window_s=0.05,
    max_gap_s=0.18,
    max_gap_fit_error_m=0.40,
    bounce_max_height_m=0.27,
    bounce_min_vertical_mps=0.75,
    bounce_min_prominence_m=0.012,
    shot_ball_only_delta_mps=12.0,
    shot_min_outgoing_mps=6.0,
    shot_min_speed_gain_mps=8.0,
    net_plane_tolerance_m=0.36,
    net_max_y_retention=0.55,
)
VARIANTS = (
    ("initial", {}),
    ("short_velocity", {"velocity_window_s": 1 / 30}),
    ("long_velocity", {"velocity_window_s": 1 / 15}),
    ("medium_gap", {"max_gap_s": 0.24}),
    ("wide_gap_then_initial", {"max_gap_s": 0.34, "max_gap_fit_error_m": 0.55}),
    (
        "wide_gap_soft_shot",
        {
            "max_gap_s": 0.34,
            "max_gap_fit_error_m": 0.55,
            "shot_ball_only_delta_mps": 9.0,
        },
    ),
    (
        "wide_gap_weak_shot",
        {
            "max_gap_s": 0.34,
            "max_gap_fit_error_m": 0.55,
            "shot_ball_only_delta_mps": 7.5,
            "shot_min_outgoing_mps": 4.5,
            "shot_min_speed_gain_mps": 6.5,
        },
    ),
    (
        "soft_bounce",
        {"bounce_min_vertical_mps": 0.45, "bounce_min_prominence_m": 0.008},
    ),
    ("low_bounce_height", {"bounce_max_height_m": 0.18}),
    ("soft_shot", {"shot_ball_only_delta_mps": 9.0}),
    ("strong_shot", {"shot_ball_only_delta_mps": 16.0}),
    ("tight_net", {"net_plane_tolerance_m": 0.22, "net_max_y_retention": 0.40}),
    ("wide_net", {"net_plane_tolerance_m": 0.5, "net_max_y_retention": 0.67}),
)


@dataclass
class Track:
    xyz: NDArray[np.float32]
    event_mask: NDArray[np.uint8]
    fps: int
    group: str


@dataclass
class Observation:
    track: Track
    xyz: NDArray[np.float32]
    valid: NDArray[np.bool_]


def simulate_net_tracks(root: Path, split: str) -> list[Track]:
    config = build_generator_config(OmegaConf.load(root / "config.yaml"))
    physics = config.physics
    physics.dt = 1 / config.rally.sim_fps
    simulator = RallySimulator(
        physics_config=physics,
        rally_config=config.rally,
        cell_manager=CellManager(),
        targeted_velocity_config=config.targeted_velocity,
        device="cpu",
    )
    stride = config.rally.sim_fps // config.rally.output_fps
    rng = np.random.default_rng(SIM_SEED[split])
    cases = []
    for profile in ("mesh", "tape", "clear"):
        count = 0
        for _ in range(384):
            x = rng.uniform(-4.8, 4.8)
            side = -1 if rng.random() < 0.5 else 1
            vx = rng.uniform(-1.0, 1.0)
            vy = rng.uniform(5.0, 8.0) if profile == "tape" else rng.uniform(7.0, 26.0)
            vz = rng.uniform(-2.0, 1.0)
            distance = rng.uniform(0.85, 1.7)
            arrival = distance / vy
            height = net_height_at_x(x + vx * arrival)
            if profile == "mesh":
                clearance = -rng.uniform(0.12, 0.60)
            elif profile == "tape":
                clearance = -rng.uniform(0.003, 0.026)
            else:
                clearance = rng.uniform(0.025, 0.34)
            z = height + clearance - vz * arrival + 0.5 * physics.gravity * arrival**2
            state = BallState(
                torch.tensor([x, side * distance, z], dtype=torch.float32),
                torch.tensor([vx, -side * vy, vz], dtype=torch.float32),
                torch.zeros(3),
            )
            # Use the same substep discontinuity recorder as simulate_rally.
            # Starting samples are in-flight and do not define a shot event.
            flight = simulator._simulate_single_shot(
                state,
                from_cell=0,
                from_side="near" if side < 0 else "far",
                max_frames=144,
                max_bounces=2,
            )
            net = next((e for e in flight["events"] if e.kind == "net"), None)
            if net is None:
                actual = "clear" if flight["t_net_sim"] >= 0 else "no_crossing"
            else:
                x_hit, _, z_hit = net.position.tolist()
                actual = (
                    "tape"
                    if z_hit > net_height_at_x(x_hit) - physics.net_cord_radius
                    else "mesh"
                )
            if profile != actual:
                continue
            xyz = torch.stack(flight["trajectory_sim"])[::stride].numpy()
            mask = np.zeros(len(xyz), np.uint8)
            for event in flight["events"]:
                frame = -(-event.step // stride)
                if frame < len(xyz):
                    mask[frame] |= EVENT_BITS[event.kind]
            group = profile
            if (
                profile == "tape"
                and net is not None
                and net.velocity_before[1] * net.velocity_after[1] > 0
            ):
                group = "tape_forward"
            cases.append(Track(xyz, mask, config.rally.output_fps, group))
            count += 1
            if count == NET_CASES_PER_GROUP:
                break
        if count != NET_CASES_PER_GROUP:
            raise ValueError(f"{split}: obtained {count} {profile} tracks")
    return cases


def observe(tracks: list[Track], split: str, *, corrupt: bool) -> list[Observation]:
    result = []
    seeds = np.random.SeedSequence(MASK_SEED[split]).spawn(len(tracks))
    for track, seed in zip(tracks, seeds, strict=True):
        frames = len(track.xyz)
        if not corrupt:
            result.append(Observation(track, track.xyz, np.ones(frames, dtype=bool)))
            continue
        events, isolated, noise = (np.random.default_rng(s) for s in seed.spawn(3))
        missing, _, _ = sample_occlusion(
            np.ones((1, frames), dtype=bool),
            track.event_mask,
            CORRUPTION,
            events,
            isolated,
        )
        xyz = (track.xyz + noise.normal(0, JITTER_M, track.xyz.shape)).astype(
            np.float32
        )
        xyz[missing[0]] = 0
        result.append(Observation(track, xyz, ~missing[0]))
    return result


def evaluate(
    samples: list[Observation], rules: RuleThresholds, kinds: tuple[EventKind, ...]
) -> dict:
    predictions = [
        detect_ball_events(s.xyz, s.valid, s.track.fps, thresholds=rules)
        for s in samples
    ]
    report = {
        "tracks": len(samples),
        "frames": sum(len(s.xyz) for s in samples),
        "missing_frames": sum(int((~s.valid).sum()) for s in samples),
        "groups": dict(Counter(s.track.group for s in samples)),
        "classes": {},
    }
    for kind in kinds:
        pred = [
            np.array([e.frame for e in events if e.kind == kind], dtype=np.int64)
            for events in predictions
        ]
        truth = [np.flatnonzero(s.track.event_mask & EVENT_BITS[kind]) for s in samples]
        metrics = event_detection_report(pred, truth, TOLERANCE_FRAMES)
        metrics["boundary_events"] = sum(
            int(((t == 0) | (t == len(sample.xyz) - 1)).sum())
            for t, sample in zip(truth, samples, strict=True)
        )
        metrics["unobserved_events"] = sum(
            int((~sample.valid[t]).sum())
            for t, sample in zip(truth, samples, strict=True)
        )
        metrics["predictions_spanning_gaps"] = sum(
            e.kind == kind and "spans_ball_gap" in e.flags
            for events in predictions
            for e in events
        )
        mae = metrics["mean_abs_offset_frames"]
        metrics["mae_ms"] = None if mae is None else 1000 * mae / samples[0].track.fps
        report["classes"][kind] = metrics
    return report


def measure_pair(
    rally: list[Observation],
    net: list[Observation],
    rules: RuleThresholds,
) -> dict:
    return {
        "rallies": evaluate(rally, rules, ("shot", "bounce")),
        "net_flights": evaluate(net, rules, ("net",)),
    }


def macro_f1(report: dict) -> float:
    return sum(
        result["classes"][kind]["f1"]
        for table, kind in (
            ("rallies", "shot"),
            ("rallies", "bounce"),
            ("net_flights", "net"),
        )
        for result in (report[table],)
    ) / len(EVENT_KINDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    options = parser.parse_args()
    if (
        sha256((options.dataset / "config.yaml").read_bytes()).hexdigest()
        != CONFIG_SHA256
    ):
        raise ValueError("BLCS configuration checksum mismatch")
    dataset = SharedDataset(options.dataset)
    if dataset.manifest_hash != MANIFEST_SHA256:
        raise ValueError("Shared dataset manifest checksum mismatch")
    torch.set_num_threads(1)
    val_rallies = [
        Track(s.xyz, s.events, s.physics.output_fps, "shared_rally")
        for s in dataset.split("val")
    ]
    val_net = simulate_net_tracks(options.dataset, "val")
    val_gap = observe(val_rallies, "val", corrupt=True)
    val_net_gap = observe(val_net, "val", corrupt=True)
    sweep = []
    for name, changes in VARIANTS:
        rules = replace(INITIAL_THRESHOLDS, **changes)
        metrics = measure_pair(val_gap, val_net_gap, rules)
        sweep.append(
            {
                "name": name,
                "thresholds": asdict(rules),
                "macro_f1": macro_f1(metrics),
                **metrics,
            }
        )
    best = max(sweep, key=lambda row: row["macro_f1"])
    result = {
        "shared_manifest_sha256": MANIFEST_SHA256,
        "blcs_config_sha256": CONFIG_SHA256,
        "simulator_seed": SIM_SEED,
        "observation_seed": MASK_SEED,
        "corruption": asdict(CORRUPTION),
        "jitter_sigma_m": JITTER_M,
        "tolerance_frames": TOLERANCE_FRAMES,
        "validation_selection": best["name"],
        "validation_sweep": sweep,
        "thresholds": best["thresholds"],
    }
    if not options.validation_only:
        if best["thresholds"] != asdict(DEFAULT_THRESHOLDS):
            raise ValueError(
                "Detector defaults differ from the selected validation thresholds"
            )
        test_rallies = [
            Track(s.xyz, s.events, s.physics.output_fps, "shared_rally")
            for s in dataset.split("test")
        ]
        test_net = simulate_net_tracks(options.dataset, "test")
        clean = measure_pair(
            observe(test_rallies, "test", corrupt=False),
            observe(test_net, "test", corrupt=False),
            DEFAULT_THRESHOLDS,
        )
        net_gap = observe(test_net, "test", corrupt=True)
        occluded = measure_pair(
            observe(test_rallies, "test", corrupt=True),
            net_gap,
            DEFAULT_THRESHOLDS,
        )
        result["test"] = {"clean": clean, "occluded": occluded}
        result["occluded_net_by_group"] = {
            group: evaluate(
                [sample for sample in net_gap if sample.track.group == group],
                DEFAULT_THRESHOLDS,
                ("net",),
            )
            for group in sorted({sample.track.group for sample in net_gap})
        }
    write_json_atomic(options.output, result)
    print("Validation selection:", best["name"], best["macro_f1"])
    print("Metrics:", options.output)


if __name__ == "__main__":
    main()
