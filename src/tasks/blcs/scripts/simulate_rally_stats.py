"""Quickly measure rally-length / end-reason distributions without cameras.

Simulates rallies with the same per-scene physics sampling as the full
dataset generator, but skips camera projection and disk IO so distribution
tuning iterates fast.

Usage:
    python -m src.tasks.blcs.scripts.simulate_rally_stats
    python -m src.tasks.blcs.scripts.simulate_rally_stats --num-rallies 200
    python -m src.tasks.blcs.scripts.simulate_rally_stats --no-refine
"""

from __future__ import annotations

import argparse
import collections
import statistics
import time

import torch

from src.tasks.blcs.generate_dataset.config import build_default_generator_config
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneGenerator
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallySimulator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rallies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable physics-based landing refinement (legacy behaviour)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = build_default_generator_config()
    if args.no_refine:
        config.targeted_velocity.landing_refine_enabled = False
        config.targeted_velocity.target_margin_m = 0.0

    generator = BLCSSceneGenerator(config=config)

    rally_lengths: list[int] = []
    end_reasons: collections.Counter[str] = collections.Counter()
    start = time.perf_counter()

    for _ in range(args.num_rallies):
        physics_config = config.physics.sample()
        simulator = RallySimulator(
            physics_config=physics_config,
            rally_config=config.rally,
            targeted_velocity_config=config.targeted_velocity,
        )
        from_cell = generator.sample_from_cell()
        side = generator.sample_side()
        result = simulator.generate_rally(from_cell, side)
        rally_lengths.append(result.rally_length)
        end_reasons[result.end_reason.value] += 1

    elapsed = time.perf_counter() - start

    print(f"rallies: {len(rally_lengths)}  ({elapsed:.1f}s, "
          f"{elapsed / max(1, len(rally_lengths)):.2f}s/rally)")
    print(f"rally_length: mean={statistics.mean(rally_lengths):.2f} "
          f"median={statistics.median(rally_lengths)} "
          f"min={min(rally_lengths)} max={max(rally_lengths)}")

    hist = collections.Counter(rally_lengths)
    total = len(rally_lengths)
    print("rally_length histogram:")
    for length in sorted(hist):
        count = hist[length]
        bar = "#" * round(50 * count / total)
        print(f"  {length:3d}: {count:5d} ({100 * count / total:5.1f}%) {bar}")

    print("end_reason distribution:")
    for reason, count in end_reasons.most_common():
        print(f"  {reason:18s}: {count:5d} ({100 * count / total:5.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
