"""Compare newly manifested shards with every available prior frozen RGB hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.distribution import load_distribution
from src.utils.checksum import dual_sha256


def check(source_repo: Path, directories: list[Path]) -> dict[str, Any]:
    pinned = json.loads(Path("knowledge/runs/run-ball-all-frame-protocol-20261022/inputs.json").read_text())
    source_manifest = source_repo / pinned["source_manifest"]
    if dual_sha256(source_manifest) != pinned["source_manifest_sha256"]:
        raise ValueError("The original frozen window manifest changed")
    legacy = json.loads(source_manifest.read_text())
    known = {c["clip"]["clip_id"]: c["rgb_shard_sha256"] for c in legacy["clips"]}
    units = {}
    for directory in directories:
        distribution = load_distribution(directory)
        if distribution["source"]["hashes"] != pinned["snapshot_hashes"]:
            raise ValueError("A distribution used a different snapshot")
        files = {f["path"]: f for f in distribution["files"]}
        matches = missing = 0
        for clip in distribution["clips"]:
            if clip["clip_id"] not in known:
                missing += 1
            elif files[clip["payload"]]["sha256"] != known[clip["clip_id"]]:
                raise ValueError("A previously frozen RGB shard changed: " + clip["clip_id"])
            else:
                matches += 1
        units[directory.name] = dict(prior_manifest_rgb_matches=matches,
                                    newly_hashed_clips_without_prior_play_window=missing,
                                    all_available_prior_rgb_hashes_match=True)
    return dict(schema_version=1, source_manifest_sha256=pinned["source_manifest_sha256"], units=units)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-root", type=Path, required=True)
    parser.add_argument("--distribution", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.source_repo_root, args.distribution), indent=2, sort_keys=True) + "\n")
