"""Add a two-method common cohort to the immutable v1 prediction experiment."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

from diagnostics import export_tables, plot_diagnostics

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import score_comparison
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.visualization.ball3d.report import (
    broken_html_references,
    plot_metrics_table,
    write_report,
)
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "annotations", "prior", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("bundle-sha256", "annotations-sha256", "prior-manifest-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            str(Path(__file__)),
            str(Path(__file__).with_name("diagnostics.py")),
        ],
        text=True,
    ):
        raise ValueError("Commit report code before recording its provenance")
    if args.output.exists():
        raise ValueError("Use a new report directory; the prior run remains immutable")
    if any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (args.prior, args.bundle, args.annotations.parent)
    ):
        raise ValueError("The final report must be separate from immutable inputs")
    if dual_sha256(args.prior / "artifacts.json") != args.prior_manifest_sha256:
        raise ValueError("Prior artifact manifest digest mismatch")
    manifest = json.loads((args.prior / "artifacts.json").read_text())
    subprocess.run(
        ["git", "diff", "--quiet", manifest["code_commit"], "HEAD", "--", "src"],
        check=True,
    )
    declared = set(manifest["files"])
    actual = {
        str(p.relative_to(args.prior)) for p in args.prior.rglob("*") if p.is_file()
    }
    if actual != declared | {"artifacts.json", "artifacts.sha256"}:
        raise ValueError("Prior has missing or undeclared files")
    for relative, entry in manifest["files"].items():
        path = args.prior / relative
        if (
            path.stat().st_size != entry["size_bytes"]
            or dual_sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"Prior checksum mismatch: {relative}")
    bundle = load_bundle(args.bundle, expected_sha256=args.bundle_sha256)
    annotations = load_annotations(
        args.annotations, bundle, expected_sha256=args.annotations_sha256
    )
    names = (
        "pipeline_baseline",
        "segment_physics_candidates",
        "segment_physics_heatmap",
        "annotation_oracle",
    )
    folds = ("all_views", *(f"without_{c}" for c in bundle.inputs.camera_ids))
    runs = {
        name: tuple(
            load_run(
                args.prior
                / (
                    "scoring_oracle"
                    if name == "annotation_oracle"
                    else f"results/{name}"
                )
                / fold
            )
            for fold in folds
        )
        for name in names
    }
    scores = score_comparison(runs, bundle, annotations)
    if scores != json.loads((args.prior / "scores.json").read_text()):
        raise ValueError("Saved scores changed during independent typed reload/rescore")
    pair = {
        name: runs[name] for name in ("pipeline_baseline", "segment_physics_candidates")
    }
    paired = score_comparison(pair, bundle, annotations)
    # Clone validated immutable predictions/media, then write additional reports.
    shutil.copytree(
        args.prior,
        args.output,
        ignore=shutil.ignore_patterns("artifacts.json", "artifacts.sha256"),
    )
    write_json_atomic(args.output / "paired-scores.json", paired)
    pair_dir = args.output / "paired"
    pair_dir.mkdir()
    pair_rows = export_tables(paired, pair_dir)
    plot_metrics_table(paired, args.output / "paired-metrics-table.png")
    plot_diagnostics(runs, bundle.inputs.fps, args.output / "physics-diagnostics.png")
    verification = json.loads((args.prior / "verification.json").read_text())
    verification.update(
        {
            "report_code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "prior_artifact_manifest_sha256": args.prior_manifest_sha256,
            "verified_prior_files": len(declared),
            "rescore_all_fields_equal": True,
            "predictions_and_video_reused_without_reexecution": True,
            "common_all_methods": "Empty because heatmap optimization failed in every fold; unavailable errors are not zero",
            "paired_common_cohort": "Baseline and candidate physics only; original metrics and failed method remain in the main table",
            "model_parameters_changed": False,
            "labels_used_for_retuning": False,
        }
    )
    page = write_report(
        args.output,
        scores,
        video=args.output / "comparison.mp4",
        figure=args.output / "trajectories.png",
        crops=args.output / "extremes.png",
        table_figure=args.output / "metrics-table.png",
        verification=verification,
    )
    extra = "<h2>Failed dense fit and comparison on common available points</h2><p>The dense heatmap fit reached the 100-evaluation limit in all four folds. Its offsets are final failed iterates, not successful estimates. Its output is invalid and its errors are unavailable. The common cohort across all four methods is empty. The additional table below compares baseline and candidate physics on their shared camera/frame points only; it does not remove the failed method from the main results.</p>"
    for title, filename in (
        ("Baseline / candidates common cohort", "paired-metrics-table.png"),
        ("Segments, physical residual and clocks", "physics-diagnostics.png"),
    ):
        png = base64.b64encode((args.output / filename).read_bytes()).decode("ascii")
        extra += (
            f'<h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{png}">'
        )
    page.write_text(
        page.read_text().replace(
            "<h2>Provenance and verification</h2>",
            extra + "<h2>Provenance and verification</h2>",
        )
    )
    verification["broken_html_references"] = broken_html_references(page)
    if verification["broken_html_references"]:
        raise ValueError("Broken final HTML references")
    verification["finalization_seconds"] = time.perf_counter() - started
    write_json_atomic(args.output / "verification.json", verification)
    small = json.loads((args.output / "metrics-summary.json").read_text())
    small["paired_rows"] = pair_rows
    write_json_atomic(args.output / "metrics-summary.json", small)
    inventory = {
        "schema": "tennis_lab_ball3d_artifact_inventory.v1",
        "code_commit": verification["code_commit"],
        "report_code_commit": verification["report_code_commit"],
        "inputs": manifest["inputs"],
        "files": {
            str(p.relative_to(args.output)): {
                "sha256": dual_sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    write_json_atomic(args.output / "artifacts.json", inventory)
    (args.output / "artifacts.sha256").write_text(
        dual_sha256(args.output / "artifacts.json") + "\n"
    )
    print(f"Validated all v1 files and scores; finalized {page}", flush=True)


if __name__ == "__main__":
    main()
