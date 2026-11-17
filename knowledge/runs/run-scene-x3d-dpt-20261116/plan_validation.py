"""Freeze a label-selected sample and dtype tolerances before any GPU inference."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = args.snapshot / "validation-update-00010000.frames.jsonl"
    groups = defaultdict(list)
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        groups[row["clip_id"]].append(row)
    manifest = json.loads((args.snapshot / "evaluation_manifest.json").read_text())
    records = {r["clip"]["clip_id"]: r for r in manifest["clips"]}
    selected = []
    for fps in ("25", "30", "50"):
        clips = [key for key, rows in groups.items() if rows[0]["native_fps"] == fps
                 and any(r["kind"] == "reviewed_empty" for r in rows)]
        clip = min(clips)
        windows = defaultdict(list)
        for row in groups[clip]:
            windows[row["owner_start"]].append(row)
        first = min(windows)
        absent = min(windows, key=lambda start: (-sum(r["kind"] in {"reviewed_empty", "out_of_frame"} for r in windows[start]), start))
        for start in sorted({first, absent}):
            selected.append(dict(clip_id=clip, native_fps=fps, start=start, metadata=records[clip]["clip"],
                                 shard_sha256=records[clip]["rgb_shard_sha256"], rows=windows[start]))
    plan = dict(schema="scene_dpt_validation_plan.v1", selection_rule="FPS25/30/50; lexicographically first clip with reviewed_empty; first and most-negative owner windows, ties earlier",
                predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                tolerance_comparison="Coordinates in source pixels, max absolute component error; score absolute error. All selected owner frames must pass; no exclusions by error/score.",
                tolerances={"bf16": dict(source_px_atol=1.0, score_atol=.005),
                            "fp32": dict(source_px_atol=2.0, score_atol=.01)},
                tolerance_reason="Historical predictions use L4 BF16. FP32-vs-BF16 includes rounding difference; same-dtype port parity is checked separately against the frozen encoder.",
                support=dict(windows=len(selected), frames=sum(len(w["rows"]) for w in selected),
                             kinds=dict(Counter(r["kind"] for w in selected for r in w["rows"]))),
                windows=selected)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan["support"]))
    for window in selected:
        print(window["native_fps"], window["metadata"]["index"], window["start"], window["shard_sha256"])


if __name__ == "__main__":
    main()
