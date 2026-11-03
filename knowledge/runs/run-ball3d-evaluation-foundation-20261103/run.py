"""clip_000-only reproduction entrypoint; reusable code lives in src."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import (
    create_bundle,
    load_bundle,
    validate_source_candidate,
    verify_videos,
)
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison, save_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    annotation_oracle,
    score_comparison,
)
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.visualization.ball3d.rendering import (
    BallComparisonRenderer,
    render_video,
)
from src.tennis_scene.visualization.ball3d.report import (
    broken_html_references,
    plot_comparison,
    render_extremes,
    write_report,
)
from src.utils.checksum import dual_sha256

CLIP_ID = "meiji_3cam/video_000/clip_000"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("bundle")
    build.add_argument("--store", type=Path, required=True)
    build.add_argument("--candidate", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--source-code-commit", required=True)
    build.add_argument("--source-code-evidence", required=True)
    build.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--bundle", type=Path, required=True)
    compare.add_argument("--annotations", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--video-step", type=int, default=1)
    compare.add_argument("--width", type=int, default=1200)
    for command in (build, compare):
        command.add_argument("--videos", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    if args.action == "bundle":
        candidate = validate_source_candidate(args.candidate, args.store)
        if candidate["source_code_commit"] != args.source_code_commit:
            raise ValueError("Candidate and supplied historical code commit disagree")
        bundle = create_bundle(args.store, args.output, clip_id=CLIP_ID,
            source_code_commit=args.source_code_commit, source_code_evidence=args.source_code_evidence,
            source_archive=args.archive)
        videos = {c: args.videos / f"{c}.mp4" for c in bundle.inputs.camera_ids}
        receipt = {"candidate": candidate, "bundle_sha256": bundle.manifest_sha256,
                   "videos": verify_videos(bundle, videos), "elapsed_seconds": time.perf_counter() - started}
        write_json_atomic(args.output.parent / f"{args.output.name}-verification.json", receipt)
        print(receipt, flush=True)
        return
    bundle = load_bundle(args.bundle)
    inputs = bundle.inputs
    if inputs.clip_id != CLIP_ID or inputs.num_frames != 1010 or inputs.camera_ids != ("cam0", "cam1", "cam2"):
        raise ValueError("This reproduction script is restricted to the agreed clip_000 inputs")
    if args.output.exists():
        raise ValueError("Use a new output directory; never overwrite a prior comparison")
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline = verify_baseline(bundle)
    print("Verified fixed inputs and baseline", flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    src = Path("src")
    fingerprint = document_digest({str(p.relative_to(src)): dual_sha256(p) for p in sorted(src.rglob("*.py"))})
    args.output.mkdir(parents=True)
    methods = {method.name: method for method in (baseline_from_bundle(bundle), baseline_from_bundle(bundle, without_loess=True))}
    runs = run_comparison(methods, inputs, args.output / "results", bundle_sha256=bundle.manifest_sha256,
                          code_commit=commit, code_fingerprint=fingerprint)
    # Load every published result through the real checksum/type boundary.
    runs = {name: tuple(load_run(args.output / "results" / name / run.key) for run in values)
            for name, values in runs.items()}
    print("Saved and reloaded all eight CPU runs", flush=True)
    annotations = None if args.annotations is None else load_annotations(args.annotations, bundle)
    if annotations is not None:
        if annotations.provenance["model"] != "gpt-6.1-sol" or annotations.provenance["effort"] != "max":
            raise ValueError("clip_000 labels must use the agreed GPT-6.1 Sol/max direct-video review")
        oracle = annotation_oracle(inputs, annotations, bundle_sha256=bundle.manifest_sha256,
                                   code_commit=commit, code_fingerprint=fingerprint)
        for run in oracle:
            save_run(args.output / "scoring_oracle" / run.key, run)
        runs["annotation_oracle"] = oracle
    scores = score_comparison(runs, inputs, annotations)
    full = {name: next(r for r in values if r.held_out_camera is None)
            for name, values in runs.items() if name != "annotation_oracle"}
    with BallComparisonRenderer(inputs, full, videos, annotations=annotations, width=args.width) as renderer:
        media = render_video(renderer, args.output / "comparison.mp4", step=args.video_step)
    print("Rendered and decoded every comparison frame", flush=True)
    plot_comparison(runs, inputs, args.output / "trajectories.png", annotations)
    extremes = render_extremes(runs, inputs, videos, args.output / "extremes.png", annotations)
    write_json_atomic(args.output / "extremes.json", extremes)
    verification = {"schema": "tennis_lab_ball3d_qualification.v1", "bundle_sha256": bundle.manifest_sha256,
        "source_code_commit": bundle.manifest["source_code_commit"], "replay_code_commit": commit,
        "replay_code_fingerprint": fingerprint, "replay_script_sha256": dual_sha256(Path(__file__)),
        "baseline": baseline, "videos": video_receipt, "comparison_video": media,
        "annotation_status": "unavailable; awaiting independent direct-video labels (#1665)" if annotations is None else "scored",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "gpu_used": False, "inference_rerun": False, "elapsed_seconds": time.perf_counter() - started,
        "human_adoption": "pending; no merge authorized"}
    page = write_report(args.output, scores, video=args.output / "comparison.mp4",
        figure=args.output / "trajectories.png", crops=args.output / "extremes.png" if extremes["frames"] else None,
        verification=verification)
    verification["broken_html_references"] = broken_html_references(page)
    verification["artifact_sha256"] = {p.name: dual_sha256(p) for p in (page, args.output / "scores.json",
        args.output / "comparison.mp4", args.output / "trajectories.png", args.output / "extremes.json")}
    write_json_atomic(args.output / "verification.json", verification)
    print(f"Comparison: {page} ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
