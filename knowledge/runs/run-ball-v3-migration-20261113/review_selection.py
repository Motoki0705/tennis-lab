"""Apply source-specific semantic evidence to the initial, retained inventory."""

import argparse
import json
from collections import Counter
from pathlib import Path

from inventory import digest, write_once

# These are decisions about exact source declarations, not a general NLP rule.
# The original file digests in selection.json bind every quoted statement.
TARGET_CONFLICTS = {
    "video_001/clip_002": (5, "including its toss, flight and identifiable post-rally visible/held positions"),
    "video_001/clip_005": (8, "remains the target when resting at the net"),
    "video_001/clip_006": (5, "手で保持されている状態"),
    "video_001/clip_009": (5, "retained during the pre-serve and after the rally"),
    "video_001/clip_011": (7, "hand-held before release"),
    "video_002/clip_000": (0, "保持中であっても独立して見える対象ボールは含む"),
    "video_002/clip_004": (5, "including the held ball"),
    "video_002/clip_010": (0, "手に保持された開始部分の対象ボールも含む"),
    "video_002/clip_011": (7, "保持・トス・打球"),
    "video_002/clip_014": (7, "after hitting the net and settling"),
    "video_002/clip_018": (5, "including visible pre-serve holding"),
    "video_002/clip_021": (5, "手で保持されたボールも"),
}

PROCESSING = {
    "video_000/clip_001": ("review", "all_frames_automatically_processed", True),
    "video_000/clip_002": ("annotation_notes", 9, "All source frames were processed"),
    "video_000/clip_003": ("review", "processed_frame_count", "frame_count"),
    "video_000/clip_011": ("review", "all_frames_image_processed", True),
    "video_001/clip_001": ("annotation_notes", 8, "全フレームをデコードして1フレーム1レコードを生成"),
    "video_001/clip_003": ("annotation_notes", 8, "全フレームをデコードして記録を作成"),
    "video_001/clip_007": ("annotation_notes", 8, "全フレームをデコードしてレコードを作成"),
    "video_002/clip_017": ("review", "review_mode", "automated_all_frame_processing_with_targeted_visual_review"),
}


def find_note(annotation, expected_index, text):
    # Camera notes can have an additional camera-specific paragraph. Resolve the
    # exact statement, retaining its actual JSON pointer rather than assuming an index.
    matches = [(i, note) for i, note in enumerate(annotation["annotation_notes"]) if text in note]
    if len(matches) != 1:
        raise ValueError(f"source evidence must occur exactly once: {text!r}")
    index, note = matches[0]
    return dict(pointer=f"/annotation_notes/{index}", text=note)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.initial.read_bytes())
    data.update(revision=2, previous_selection_sha256=digest(args.initial))
    decisions, reasons, counts = Counter(), Counter(), Counter()
    for entry in data["entries"]:
        annotation_path = args.source_root / data["legacy_root"] / entry["clip"] / "outsource" / f"{entry['camera']}_annotations.json"
        assert digest(annotation_path) == entry["original_sha256"]
        annotation = json.loads(annotation_path.read_bytes())
        clip = entry["key"].rsplit("/", 1)[0]
        held = []
        if annotation["target"]["scope"] != "main-court rally ball":
            held.append("target_includes_held_ball")
            entry["target_conflict_evidence"] = dict(pointer="/target/scope", text=annotation["target"]["scope"])
        if clip in TARGET_CONFLICTS:
            index, text = TARGET_CONFLICTS[clip]
            entry["target_conflict_evidence"] = find_note(annotation, index, text)
            held.append("target_includes_held_pre_serve_or_resting_ball")
        processing = None
        if annotation["summary"].get("all_frames_reviewed") is True:
            processing = "retained original #/summary/all_frames_reviewed=true"
        elif clip in PROCESSING:
            area, key, expected = PROCESSING[clip]
            if area == "annotation_notes":
                proof = find_note(annotation, key, expected)
                processing = f"retained original #{proof['pointer']}: {proof['text']}"
            else:
                value = annotation[area][key]
                if expected == "frame_count":
                    assert type(value) is int and value == annotation["source"]["frame_count"]
                elif expected is True:
                    assert value is True
                else:
                    assert value == expected
                processing = f"retained original #/{area}/{key}={value}"
        if processing is None:
            held.append("processing_completion_unconfirmed_without_missing_review_report")
        entry["processing_evidence"] = processing
        entry["reasons"] = held
        entry["decision"] = "hold" if held else "convert"
        decisions[entry["decision"]] += 1
        reasons.update(held)
        if not held:
            counts.update(entry["counts"])
    data["decisions"] = dict(decisions)
    data["selected_counts"] = dict(counts)
    data["hold_reason_counts"] = dict(reasons)
    write_once(args.output, data)
    print(json.dumps(dict(decisions=decisions, reasons=reasons, counts=counts, frames=sum(counts.values()))))


if __name__ == "__main__":
    main()
