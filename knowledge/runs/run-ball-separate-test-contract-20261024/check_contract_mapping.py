"""Check the final mapping guards against the already byte-verified real contract."""

import argparse
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.checkpoints.temporal_heatmap import save_json
from src.tasks.ball_detection.data.distribution import (
    load_distribution,
    source_payloads,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.evaluation.temporal_heatmap_contract import (
    CONTRACT_SCHEMA,
    _clip_identity,
    _pinned,
)
from src.utils.checksum import dual_sha256


def check(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Use a new mapping-check output")
    contract = _pinned(args.contract, args.contract_sha256)
    if contract["schema"] != CONTRACT_SCHEMA:
        raise ValueError("Unknown contract schema")
    results: dict[str, Any] = {}
    for name in ("training", "test"):
        root = getattr(args, name + "_distribution")
        distribution = load_distribution(root)
        if dual_sha256(root / "distribution.json") != contract[name]["distribution_sha256"]:
            raise ValueError("Pinned distribution changed")
        store = BallFrameStore(root, external_payloads=source_payloads(root, args.source_store))
        actual = _clip_identity(store, distribution)
        if (actual != contract[name]["clips_sha256"]
                or any(dual_sha256(root / n) != v for n, v in contract[name]["snapshot_hashes"].items())):
            raise ValueError("Final mapping/source semantic checks disagree with the verified contract")
        results[name] = dict(clips=len(store.clips), clips_sha256=actual, mapping_and_semantics_equal=True)
    save_json(args.output, dict(schema_version=1, status="verified_final_mapping_guards",
        contract_sha256=args.contract_sha256, comparisons=results, rgb_payloads_rehashed=False,
        contract_module_sha256=dual_sha256(Path("src/tasks/ball_detection/evaluation/temporal_heatmap_contract.py"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-store", "training-distribution", "test-distribution", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    check(parser.parse_args())
