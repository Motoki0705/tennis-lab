"""Re-derive per-run statistics (fixing h:mm:ss vs m:ss parsing) and aggregate them."""

from __future__ import annotations

import json
from pathlib import Path

EXTRA = Path("/tmp/tcd_work/extra")
summary = json.loads((EXTRA / "sweep_results.json").read_text())


def parse_wall(raw: str) -> float:
    parts = raw.strip().split(":")
    values = [float(part) for part in parts]
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def log_stats(log_path: str) -> dict[str, object]:
    text = Path(log_path).read_text()
    out: dict[str, object] = {}
    for line in text.splitlines():
        if "Elapsed (wall clock) time" in line:
            out["wall_s"] = parse_wall(line.split(": ", 1)[1])
        if "User time (seconds)" in line:
            out["user_s"] = float(line.split(": ", 1)[1])
        if "Maximum resident set size" in line:
            out["max_rss_kb"] = int(line.split(": ", 1)[1])
        if line.startswith("Reading frame with index"):
            out["middle_frame_index"] = int(line.rsplit(" ", 1)[1])
        if line.startswith("Video properties"):
            out["video_properties"] = line
        if line.startswith("Processing error:"):
            out["processing_error"] = line
        if "Assertion" in line or "terminate called" in line or "what()" in line:
            out.setdefault("abort_message", line.strip())
    return out


for record in summary["records"]:
    record["log_stats"] = log_stats(record["log"])

print(json.dumps(summary["records"], ensure_ascii=False, indent=2))
