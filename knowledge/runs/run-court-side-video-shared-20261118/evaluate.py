"""CPU build/load, historical score replay, and independent per-frame reduction on meiji_4cam."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tasks.court_side.hypothesis import distinct_observation_frames, pair_frame_counts, score_hypothesis
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.tennis_scene.pipeline.video_court_side import PinnedSideInputs, VideoCourtSideArtifact
from src.tennis_scene.pipeline.video_court_side.component import SharedCourtSideModule
from src.utils.checksum import dual_sha256

from collect_prefix import COURT_DIGEST, DRIFT_DIGEST

BATCH_COMMIT = "9a727005fd735dbdc76417cb54d476beb81cd2b3"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def turns(h):
    return "".join("T" if v else "F" for v in h.view_half_turns)


def evaluate(root, run):
    started = time.monotonic()
    completed = json.loads((run / "prefix-results.json").read_text())
    health = json.loads((run / "health/prefix.json").read_text())
    assert health["status"] == "complete" and not (run / "anomaly.json").exists()
    dataset_file = Path("data/tennis_multiview/processed/meiji_4cam/dataset/dataset.json")
    dataset = json.loads(dataset_file.read_text())
    assert dataset["dataset_id"] == "meiji_4cam" and len(completed) == 69
    assert set(completed) == {r["clip_id"] for r in dataset["clips"] if r["video_id"] == "video_000"}
    pins = {clip: PinnedSideInputs.from_store(root / entry["directory"]) for clip, entry in completed.items()}
    request = run / "side-inputs.json"
    write_json_atomic(request, {"schema": "video_court_side_inputs_v1", "evidence_clip_ids": None, "clips": {
        clip: {"role": "artifact", "directory": completed[clip]["directory"], "inputs_digest": pin.digest} for clip, pin in pins.items()
    }})
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("src/tennis_scene/configs").resolve())):
        cfg = compose(config_name="pipeline_x3d_dpt", overrides=[
            f"paths.project_root={root.parent}", f"paths.artifact_root={root}", f"paths.output_root={run / 'entry-results'}",
            "court_kp.mode=video_shared", "court_kp.video_shared.action=load",
            "court_kp.video_shared.dataset_id=meiji_4cam", "court_kp.video_shared.video_id=video_000",
            "court_kp.video_shared.directory=court-video-shared/tennis_scene/evaluate/court-video-shared-acceptance-20261117/store",
            "court_kp.video_shared.drift_directory=court-video-shared/tennis_scene/evaluate/court-video-drift-lines-20261117/drift-store",
            f"court_kp.video_shared.artifact_digest={COURT_DIGEST}", f"court_kp.video_shared.drift_digest={DRIFT_DIGEST}",
            "court_side.mode=video_shared", "court_side.video_shared.action=build",
            f"court_side.video_shared.directory={run.relative_to(root)}/side-store",
            f"court_side.video_shared.request={request.relative_to(root)}",
            f"court_side.video_shared.request_sha256={dual_sha256(request)}",
        ])
    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
    checkpoint = dual_sha256(runtime.ball_detection.checkpoint)
    settings = json_value(runtime.processing_settings["ball_detection"])
    assert settings["model_type"] == "x3d_csn_dpt"
    pipeline = TennisSceneOrchestrator(runtime)
    court = pipeline.prepare_video_court()
    side = pipeline.prepare_video_court_side(court)
    loaded = VideoCourtSideArtifact.load(side.directory, side.digest)
    assert loaded == side and set(side.clips) == set(pins)
    d = side.decision
    assert set(d.evidence_clip_ids) == set(pins)
    history_ref = BATCH_COMMIT + ":knowledge/runs/run-i1752-doubles-20261118/court-side-failures.json"
    previous = json.loads(subprocess.check_output(["git", "show", history_ref], text=True))
    ranks, clips, uv_parts, masks = [], [], [], []
    for clip, pin in sorted(pins.items()):
        index = json.loads((pin.directory / "scene.json").read_text())
        for camera in pin.source.camera_ids:
            for stage in ("ball_detection", "ball_points"):
                node = f"{stage}/{camera}"
                desc = read_component_descriptor(pin.directory, index["artifacts"][node], node=node, source_sha256=pin.source_digest)
                config = desc["identity"]["settings"]
                if stage == "ball_detection":
                    assert config["config"] == settings and config["checkpoint"]["sha256"] == checkpoint
                else:
                    assert config["smoothing"] == json_value(runtime.ball_smoothing)
        inputs = pin.assemble()
        assert all(c["drift_digest"] == court.drift_manifest["cameras"][c["camera_id"]]
                   and c["unobservable_policy"] == "warn" for c in inputs.calibration.court.diagnostics["shared_courts"])
        grid, uv, visible = inputs.sample(runtime.sampling_max_frames)
        scaled = inputs.scaled_config(runtime.court_side)
        keep = distinct_observation_frames(uv, visible, scaled.min_motion_px)
        uv_parts.append(uv[:, keep])
        masks.append(visible[:, keep])
        c = loaded.clips[clip]
        assert tuple(grid[keep & (visible.sum(axis=0) >= 2)]) == c.scored_frame_indices
        verdict = loaded.clip_report(clip, "warn")
        module = SharedCourtSideModule(pin.source.camera_ids, loaded, court, "warn")
        try:
            output = module.process(inputs)
        except ReconstructionUnavailable as failure:
            assert verdict["status"] == "failed" and failure.reason == verdict["reason"]
        else:
            assert verdict["status"] in {"ok", "warning"} and output == loaded.output()
        a = c.standalone
        if clip.endswith("/clip_000"):
            original = json.loads((pin.directory / index["artifacts"]["court_side"]["path"]).read_text())["payload"]
            assert a.reason is None and original["hypotheses"] == json_value(a.hypotheses) and original["margin"] == a.margin
        if clip in previous:
            old = previous[clip]
            assert a.hypotheses and json_value(a.hypotheses) == old["diagnostics"]["hypotheses"]
            assert "court_side_" + a.reason == old["reason"] and a.pair_frames == old["diagnostics"]["pair_frames"]
        if a.reason is None:
            classification = "decided" if d.shared.reason else ("agreement" if a.hypotheses[0].view_half_turns == loaded.output().view_half_turns else "conflict")
        else:
            classification = a.reason
        clips.append({"clip_id": clip, "source_frames": pin.source.num_frames, "sampled": c.sampled_frames, "scored": a.frames,
                      "standalone": classification, "best": turns(a.hypotheses[0]) if a.hypotheses else None,
                      "cost": a.hypotheses[0].cost if a.hypotheses else None, "support": a.hypotheses[0].support if a.hypotheses else None,
                      "runner_up": turns(a.hypotheses[1]) if a.hypotheses else None, "margin": a.margin,
                      "node_status": verdict["status"], "reason": verdict["reason"]})
        ranks.extend({"clip_id": clip, "rank": n, "turns": turns(h), "cost": h.cost, "support": h.support,
                      "frames": h.frames} for n, h in enumerate(a.hypotheses, 1))
    # Independent reduction: retain clip boundaries for the motion filter, then score the
    # concatenated distinct frame columns directly with the original per-frame DLT.
    uv_all = np.concatenate(uv_parts, axis=1)
    visible_all = np.concatenate(masks, axis=1)
    cams = tuple(v.camera for v in inputs.calibration.calibration.views)
    pairs = pair_frame_counts(visible_all)
    errors = []
    for i, cam in enumerate(d.camera_ids):
        for j in range(i + 1, len(d.camera_ids)):
            assert pairs[i, j] == d.shared.pair_frames[f"{cam}-{d.camera_ids[j]}"]
    for h in d.shared.hypotheses:
        direct = score_hypothesis(cams, h.view_half_turns, uv_all, visible_all, scaled)
        assert direct.frames == h.frames
        errors.append(max(abs(direct.cost - h.cost), abs(direct.support - h.support)))
    assert max(errors) < 1e-12
    write_csv(run / "clips.csv", clips)
    write_csv(run / "standalone-rankings.csv", ranks)
    counts = dict(Counter(row["standalone"] for row in clips))
    summary = {"dataset_id": d.dataset_id, "video_id": d.video_id, "dataset_sha256": dual_sha256(dataset_file),
               "camera_ids": d.camera_ids, "reference_camera": d.reference_camera, "clips": len(clips),
               "source_frames_per_camera": sum(row["source_frames"] for row in clips), "sampled": sum(row["sampled"] for row in clips),
               "court_digest": COURT_DIGEST, "drift_digest": DRIFT_DIGEST, "calibration_digest": d.calibration_digest,
               "checkpoint_sha256": checkpoint, "side_digest": side.digest, "diagnostics_digest": d.diagnostics_digest,
               "request_sha256": dual_sha256(request), "code_identity": pipeline.code_identity, "thresholds": d.thresholds,
               "max_frames_per_clip": d.max_frames, "unobservable_policy": "warn", "shared": json_value(d.shared),
               "standalone_counts": counts, "node_counts": dict(Counter(row["node_status"] for row in clips)),
               "independent_max_abs_cost_or_support_difference": max(errors), "historical_rankings_exact": ["clip_000", "clip_027", "clip_037"],
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "evaluation_seconds": time.monotonic() - started}
    write_json_atomic(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.outputs.resolve(), args.run.resolve())
