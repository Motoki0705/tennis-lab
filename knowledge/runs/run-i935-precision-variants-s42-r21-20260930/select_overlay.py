"""Apply the detector-only hard-clip rule posted before rendering in #935."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner.data.targets import TargetReason
from src.utils.checksum import dual_sha256

BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    plan = json.loads((BUNDLE / "plan.json").read_text())
    reference = Path(plan["reference"])
    manifest = json.loads((reference / "manifest.json").read_text())
    grouped: dict[str, list[Any]] = {}
    hashes = {}
    for entry in manifest["artifacts"]:
        if entry["source"] != "meiji" or entry["method"] != "new_detector" or entry["condition"] != "observed":
            continue
        assert entry["clip_id"].startswith("meiji/video_000/")
        path = reference / entry["path"]
        assert dual_sha256(path) == entry["sha256"]
        hashes[str(path)] = entry["sha256"]
        with np.load(path, allow_pickle=False) as z:
            wrong = (z["target_reason"] == TargetReason.OBSERVED) & (z["error_px"] > 20)
        grouped.setdefault(entry["clip_id"].rsplit("/", 1)[0], []).append(wrong)
    ranking: list[dict[str, Any]] = []
    for clip, masks in grouped.items():
        assert len(masks) == 3 and len({len(x) for x in masks}) == 1
        counts = np.stack(masks).sum(0)
        length = min(270, len(counts))
        sums = np.convolve(counts, np.ones(length, dtype=np.int64), mode="valid")
        start = int(sums.argmax())
        ranking.append({"clip": clip, "frames": len(counts), "wrong_frame_camera_count": int(counts.sum()),
                        "start": start, "length": length, "interval_wrong_count": int(sums[start])})
    ranking.sort(key=lambda x: (-x["wrong_frame_camera_count"], x["clip"]))
    selected = next(x for x in ranking if not x["clip"].endswith("/clip_010"))
    result = {"rule_issue_comment": "https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5903921668",
              "rule": "exclude clip010; maximize e9 >20px observed frame-camera count over 3 cams; lexical tie; within clip maximize count over 270 contiguous frames, earliest tie",
              "threshold_px": 20, "selected": selected, "ranking": ranking, "input_sha256": hashes}
    (BUNDLE / "clip_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(selected))


if __name__ == "__main__":
    main()
