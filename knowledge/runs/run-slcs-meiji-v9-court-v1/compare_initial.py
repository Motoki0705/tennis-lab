"""CPU-only exact comparison of three v9 calibration clips against probe A/C.

This reads completed artifacts only; it never runs a model, repairs a receipt,
updates a cache, or checks propagation to the other 53 clips. JSON matrices and
ROIs are compared using NumPy's inferred dtype; NPZ dtypes are preserved verbatim.
A new report directory receives evidence even when comparison or hashing fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.checksum import dual_sha256

MAIN = Path("/home/kamimura/projects/tennis-lab")
CLIPS = ("video_000/clip_000", "video_001/clip_003", "video_002/clip_017")
PIN = "b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383"
SETTINGS = {
    "checkpoint": "court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt",
    "samples_per_clip": 9,
    "min_score": 0.15,
    "min_points": 10,
    "ransac_px": 20.0,
    "max_fit_error_px": 15.0,
    "ball_crop_margins": {"cam0": 0.25, "cam1": 0.25, "cam2": None},
}
SAMPLE_KEYS = ("keypoints_px", "scores", "frame_indices")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def array_comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    same_shape = actual.shape == expected.shape
    return {
        "exact": bool(
            actual.dtype == expected.dtype
            and same_shape
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.array_equal(actual, expected)
        ),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "different_elements": int(np.count_nonzero(actual != expected))
        if same_shape
        else None,
    }


class Audit:
    def __init__(self) -> None:
        self.before: dict[str, str] = {}
        self.checks: dict[str, dict[str, Any]] = {}

    def digest(self, path: Path) -> str:
        key = str(path.resolve())
        if key not in self.before:
            self.before[key] = dual_sha256(path)
        return self.before[key]

    def json(self, path: Path) -> dict[str, Any]:
        self.digest(path)
        value = json.loads(path.read_text())
        require(isinstance(value, dict), f"Expected object: {path}")
        return cast(dict[str, Any], value)

    def arrays(self, path: Path) -> dict[str, np.ndarray]:
        self.digest(path)
        with np.load(path, allow_pickle=False) as saved:
            return {k: saved[k].copy() for k in saved.files}

    def compare(self, label: str, actual: np.ndarray, expected: np.ndarray) -> None:
        require(label not in self.checks, f"Duplicate comparison: {label}")
        self.checks[label] = array_comparison(actual, expected)

    def samples(self, label: str, actual: Path, expected: Path) -> None:
        a, b = self.arrays(actual), self.arrays(expected)
        require(
            set(a) == set(b) == set(SAMPLE_KEYS), f"Unexpected sample schema: {label}"
        )
        for key in SAMPLE_KEYS:
            self.compare(f"{label}/{key}", a[key], b[key])
        require(
            a["keypoints_px"].shape == (9, 14, 2)
            and a["scores"].shape == (9, 14)
            and a["frame_indices"].shape == (9,),
            f"Expected nine CourtKP14 frames: {label}",
        )


def compare_view(
    audit: Audit,
    *,
    label: str,
    camera: str,
    production: Path,
    probe: Path,
    old: Path,
    old_h: np.ndarray,
    final_h: np.ndarray,
    final_roi: list[int],
    initial_roi: list[int],
) -> None:
    for stage, variant in (("initial", "A_margin025"), ("refined", "C_savedH_union20")):
        receipt = audit.json(production / f"{camera}_court_{stage}.json")
        trial = audit.json(probe / f"{variant}.json")
        require(
            receipt["camera_id"] == camera
            and receipt["pass"] == stage
            and receipt["status"] == "fit_returned",
            f"Production pass identity: {label}/{stage}",
        )
        require(
            trial["clip"] == label
            and trial["camera"] == camera
            and trial["variant"] == variant
            and trial["status"] == "fit_returned",
            f"Probe pass identity: {label}/{variant}",
        )
        if stage == "initial":
            require(
                trial.get("repeatability_established") is True,
                f"Probe baseline repeatability not established: {label}/{camera}",
            )
        prefix = f"{label}/{camera}/{stage}"
        audit.samples(
            prefix,
            production / f"{camera}_court_{stage}_samples.npz",
            probe / f"{variant}_raw.npz",
        )
        audit.compare(
            f"{prefix}/homography",
            np.asarray(receipt["homography"]),
            np.asarray(trial["homography"]),
        )
        audit.compare(
            f"{prefix}/roi",
            np.asarray(receipt["source_roi_xyxy"]),
            np.asarray(trial["roi_xyxy"]),
        )
        audit.compare(
            f"{prefix}/published_homography",
            np.asarray(receipt["homography"]),
            old_h if stage == "initial" else final_h,
        )
        audit.compare(
            f"{prefix}/published_roi",
            np.asarray(receipt["source_roi_xyxy"]),
            np.asarray(initial_roi if stage == "initial" else final_roi),
        )
        if stage == "refined":
            audit.compare(
                f"{prefix}/initial_homography",
                np.asarray(receipt["initial_homography"]),
                old_h,
            )
            audit.samples(
                f"{prefix}/published_samples",
                production / f"{camera}_court_samples.npz",
                production / f"{camera}_court_refined_samples.npz",
            )
        else:
            audit.samples(
                f"{prefix}/old_samples",
                production / f"{camera}_court_initial_samples.npz",
                old / f"{camera}_court_samples.npz",
            )


def compare_all(
    audit: Audit, *, source: Path, target: Path, probe: Path, dataset: Path
) -> None:
    audit.digest(Path(__file__))
    runtime = audit.json(probe / "runtime.json")
    require(
        runtime["settings"] == SETTINGS
        and runtime["status"] == "diagnostic_complete_pending_image_review"
        and runtime["subpixel_refine"] is True
        and runtime["peak_threshold"] == 0.15,
        "Unexpected probe configuration/status",
    )
    probe_before, probe_after = (
        audit.json(probe / "inputs_before.json"),
        audit.json(probe / "inputs_after.json"),
    )
    require(probe_before == probe_after, "Probe inputs changed during execution")
    checkpoint = MAIN / "outputs" / str(SETTINGS["checkpoint"])
    require(
        audit.digest(checkpoint) == PIN
        and probe_before[str(checkpoint)]["dual_sha256"] == PIN,
        "Court checkpoint pin mismatch",
    )
    for clip_id in CLIPS:
        old, new = source / clip_id, target / clip_id
        previous, current = (
            audit.json(old / "court.json"),
            audit.json(new / "court.json"),
        )
        require(
            previous["identity"]["settings"] == SETTINGS
            and current["identity"]["settings"]
            == {**SETTINGS, "crop_refinement_padding_px": 20.0},
            f"Court settings: {clip_id}",
        )
        require(
            current["identity"]
            == {**previous["identity"], "settings": current["identity"]["settings"]},
            f"Changed production input identity: {clip_id}",
        )
        require(
            current["identity"]["checkpoint_sha256"] == PIN,
            f"Court checkpoint receipt: {clip_id}",
        )
        video_id, clip_name = clip_id.split("/")
        clip_root = dataset / "videos" / video_id / "clips" / clip_name
        manifest_path = clip_root / "clip.json"
        manifest = audit.json(manifest_path)
        require(
            manifest["clip_id"] == clip_id
            and manifest["camera_ids"] == ["cam0", "cam1", "cam2"],
            f"Calibration manifest: {clip_id}",
        )
        require(
            current["identity"]["clip_sha256"]
            == audit.digest(manifest_path)
            == probe_before[str(manifest_path)]["dual_sha256"],
            f"Clip input identity: {clip_id}",
        )
        for camera, relative in zip(
            manifest["camera_ids"], manifest["video_paths"], strict=True
        ):
            media = clip_root / relative
            ball = clip_root / "outsource" / f"{camera}_annotations.json"
            require(
                current["identity"]["video_sha256"][camera] == audit.digest(media),
                f"Media identity: {clip_id}/{camera}",
            )
            require(
                current["identity"]["ball_annotation_sha256"][camera]
                == audit.digest(ball),
                f"Ball identity: {clip_id}/{camera}",
            )
            if camera != "cam2":
                require(
                    probe_before[str(media)]["dual_sha256"] == audit.digest(media)
                    and probe_before[str(ball)]["dual_sha256"] == audit.digest(ball),
                    f"Probe input identity: {clip_id}/{camera}",
                )
        for name in ("court.json", "court.npz"):
            require(
                probe_before[str(old / name)]["dual_sha256"]
                == audit.digest(old / name),
                f"Old artifact changed since probe: {old / name}",
            )
        a, b = audit.arrays(new / "court.npz"), audit.arrays(old / "court.npz")
        require(
            set(a) == set(b) == {"keypoints", "homographies"},
            f"Court NPZ schema: {clip_id}",
        )
        require(
            a["homographies"].shape == b["homographies"].shape == (3, 3, 3),
            f"Court H layout: {clip_id}",
        )
        require(
            a["keypoints"].shape
            == b["keypoints"].shape
            == (3, manifest["num_frames"], 14, 2),
            f"Court KP layout: {clip_id}",
        )
        require(
            [d["camera_id"] for d in current["diagnostics"]]
            == [d["camera_id"] for d in previous["diagnostics"]]
            == ["cam0", "cam1", "cam2"],
            f"Court camera ordering: {clip_id}",
        )
        for ci, camera in enumerate(("cam0", "cam1")):
            compare_view(
                audit,
                label=clip_id,
                camera=camera,
                production=new,
                probe=probe / clip_id / camera,
                old=old,
                old_h=b["homographies"][ci],
                final_h=a["homographies"][ci],
                final_roi=current["diagnostics"][ci]["source_roi_xyxy"],
                initial_roi=previous["diagnostics"][ci]["source_roi_xyxy"],
            )
        audit.samples(
            f"{clip_id}/cam2",
            new / "cam2_court_samples.npz",
            old / "cam2_court_samples.npz",
        )
        audit.compare(
            f"{clip_id}/cam2/homography", a["homographies"][2], b["homographies"][2]
        )
        audit.compare(f"{clip_id}/cam2/keypoints", a["keypoints"][2], b["keypoints"][2])
        require(
            current["diagnostics"][2] == previous["diagnostics"][2],
            f"Cam2 diagnostics changed: {clip_id}",
        )
        indices = np.unique(
            np.linspace(0, manifest["num_frames"] - 1, 9).round().astype(int)
        )
        for camera in ("cam0", "cam1", "cam2"):
            audit.compare(
                f"{clip_id}/{camera}/expected_frame_indices",
                audit.arrays(new / f"{camera}_court_samples.npz")["frame_indices"],
                indices,
            )


def run(
    *, source: Path, target: Path, probe: Path, dataset: Path, output: Path
) -> None:
    require(
        output.is_absolute() and not output.exists(),
        "Report directory must be new and absolute",
    )
    require(
        all(
            not output.is_relative_to(path) and not path.is_relative_to(output)
            for path in (source, target, probe, dataset)
        ),
        "Report must be separate from inputs",
    )
    output.mkdir(parents=True)
    audit = Audit()
    failure: str | None = None
    after: dict[str, str] = {}
    try:
        compare_all(audit, source=source, target=target, probe=probe, dataset=dataset)
        require(
            all(check["exact"] for check in audit.checks.values()),
            "Exact comparison failed",
        )
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        hash_errors = {}
        for name in audit.before:
            try:
                after[name] = dual_sha256(Path(name))
            except Exception as error:
                hash_errors[name] = repr(error)
        stable = not hash_errors and audit.before == after
        receipt = {
            "status": "passed" if failure is None and stable else "failed",
            "error": failure,
            "clips": list(CLIPS),
            "source": str(source),
            "target": str(target),
            "probe": str(probe),
            "checkpoint_pin": PIN,
            "checks": audit.checks,
            "inputs_before": audit.before,
            "inputs_after": after,
            "inputs_stable": stable,
            "hash_errors": hash_errors,
        }
        with (output / "comparison.json").open("x") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if failure is None:
            require(stable, "Input hashing changed or failed; see comparison.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "probe", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=MAIN / "data/tennis_multivew/processed/meiji_3cam/dataset",
    )
    args = parser.parse_args()
    run(
        **{
            name: getattr(args, name).resolve()
            for name in ("source", "target", "probe", "dataset", "output")
        }
    )


if __name__ == "__main__":
    main()
