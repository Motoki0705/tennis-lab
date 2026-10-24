"""Freeze a 192-frame, framing-stratified QA probe; never refill after rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from plan_variants import variants
from support import GROUPS, SCENES, write

from src.synthetic_data_generation.dataset.court.scale.core import digest

PROBE_VARIANTS = ("current", "wide125", "hull25_margin0", "hull25_margin0_wide125")


def select(rows: list[dict[str, Any]], scene: str) -> dict[str, Any]:
    eligible = [r for r in rows if r["scene"] == scene and r["geometry_pass"]]
    pools = {
        g: sorted(
            (r for r in eligible if r["framing"] == g),
            key=lambda r: hashlib.sha256(
                f"d2-probe-s42:{scene}:{r['pair_id']}".encode()
            ).hexdigest(),
        )
        for g in GROUPS
    }
    allocation = {g: min(4, len(pool)) for g, pool in pools.items()}
    for g in ("partial", "full", "near_full"):
        allocation[g] += min(
            12 - sum(allocation.values()), len(pools[g]) - allocation[g]
        )
    assert sum(allocation.values()) == 12
    selected: list[dict[str, Any]] = []
    strata = {}
    for g, pool in pools.items():
        n = allocation[g]
        strata[g] = {
            "geometry_population": len(pool),
            "sampled": n,
            "population_per_sample": len(pool) / n if n else None,
        }
        selected.extend(
            {
                k: row[k]
                for k in (
                    "sample_id",
                    "pair_id",
                    "ordinal",
                    "framing",
                    "planned_record_sha256",
                )
            }
            for row in pool[:n]
        )
    assert len({r["sample_id"] for r in selected}) == 12
    return {"scene": scene, "strata": strata, "samples": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Frozen probe selection already exists")
    scenarios = []
    for name in PROBE_VARIANTS:
        rows = json.loads((args.plans / name / "rows.json").read_text())
        for scene in SCENES:
            scenarios.append({"variant": name, **select(rows, scene)})
    payload = {
        "schema": "court_d2_render_probe_selection_v1",
        "generation_seed": 42,
        "trajectory_count_per_scene": 128,
        "density_level": 0,
        "rule": "in_frame_kp14_v1",
        "per_scene_per_variant": 12,
        "maximum_total_render_attempts": 200,
        "planned_render_frames": sum(len(s["samples"]) for s in scenarios),
        "sampling": "four per geometric full/near_full/partial stratum; shortages reassigned to partial then full then near_full; SHA256 order over seed42 namespace, scene and comparison_pair_id",
        "analysis": "Report raw n/acceptance/reasons by stratum; use N_h/n_h for population-weighted descriptive rates. The raw balanced sample is not a population acceptance rate. No QA-based refill or extra rendering.",
        "retry": "No automatic retry or fresh-root restart. Failed/unknown renderer calls consume their submitted camera count; preserve the attempted-frame ledger and consult the remaining stage budget before any continuation.",
        "recipes": {name: variants()[name].to_dict() for name in PROBE_VARIANTS},
        "scenarios": scenarios,
    }
    assert payload["planned_render_frames"] == 192 <= 200
    write(args.output, {**payload, "selection_sha256": digest(payload)})
    print(
        json.dumps(
            {
                "selection_sha256": digest(payload),
                "planned_render_frames": payload["planned_render_frames"],
            }
        )
    )


if __name__ == "__main__":
    main()
