"""Compare public NHT camera outputs with the completed version's render counts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.utils.data.image_record_store import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--initial-completion", type=Path, required=True)
    parser.add_argument("--dataset-version", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text()
    decoder = json.JSONDecoder()
    rendered: list[str] = []
    calls: Counter[str] = Counter()
    frames: Counter[str] = Counter()
    for start in re.finditer(r'\{\s*"schema": "nht_render_result_v1"', text):
        result, _ = decoder.raw_decode(text[start.start() :])
        scene_id = result["scene_id"]
        calls[scene_id] += 1
        ids = [item["camera_id"] for item in result["renders"]]
        frames[scene_id] += len(ids)
        rendered.extend(ids)
    assert rendered and len(rendered) == len(set(rendered))
    initial = json.loads(args.initial_completion.read_text())
    initial_ids = set(initial["member_ids"])
    assert not initial_ids.intersection(rendered)
    version = json.loads(args.dataset_version.read_text())
    expected = {
        scene["scene_id"]: scene["counts"]["rendered_unique"]
        for scene in version["scenes"]
    }
    assert dict(frames) == expected
    result = {
        "schema": "court_scale_renderer_log_verification_v1",
        "log_sha256": file_sha256(args.log),
        "dataset_version_sha256": file_sha256(args.dataset_version),
        "initial_completion_sha256": file_sha256(args.initial_completion),
        "initial_cache_member_count": len(initial_ids),
        "initial_ids_redrawn": 0,
        "unique_new_rendered_ids": len(rendered),
        "public_renderer_calls_by_scene": dict(calls),
        "new_rendered_frames_by_scene": dict(frames),
        "matches_dataset_rendered_unique": True,
        "duplicate_new_rendered_ids": 0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
