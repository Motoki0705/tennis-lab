"""Verify saved FastHMR conversion evidence and the fixed review mesh outputs."""

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

JOBS = {
    "trt": "0-8204434740687790252_2026-11-25T00-07-39-312209747Z_2393387_i2491-fasthmr-trt16-export",
    "review": "0-8204421172867557512_2026-11-25T03-53-47-132442487Z_31336_i2491-fasthmr-pt16-visual-6clips",
}
RUN_HEAD = "97cabfef23bdab3e1097af32a2cdd96cf754b9dd"
SHARED = (
    "source_frames",
    "camera_ids",
    "player_ids",
    "selected_view_index",
    "selected_view_observed",
    "segment_index",
    "K",
    "R",
    "t",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_runs(queue):
    paths = {kind: queue / "repro" / stem for kind, stem in JOBS.items()}
    for kind, path in paths.items():
        run = json.loads((path / "run.json").read_text())
        if (
            run["commit"] != RUN_HEAD
            or not (queue / "done" / f"{JOBS[kind]}.job").is_file()
        ):
            raise ValueError(f"Missing terminal run evidence for {kind}")
    return paths


def verify(queue, files):
    paths = read_runs(queue)
    export = paths["trt"] / "predictions"
    report = export / "trt-export.json"
    reference = files.parent / "run-fasthmr-camerahmr-trt-export-probe" / report.name
    if report.read_bytes() != reference.read_bytes():
        raise ValueError("Conversion receipt differs from the archived record")
    diagnostic = json.loads(report.read_text())
    if (
        diagnostic["trt_engine"] != "not_built"
        or any(
            diagnostic[key]["status"] != "failed"
            for key in ("torch_export", "torchscript_export")
        )
        or any(
            p.suffix in {".onnx", ".plan", ".engine"} for p in paths["trt"].rglob("*")
        )
    ):
        raise ValueError("Unexpected conversion outputs")

    inputs = paths["review"] / "predictions"
    axes, frames, observed, mesh_count = {}, 0, set(), {}
    for mode in ("gvhmr-archive", "pt16"):
        report = inputs / f"review-{mode}.json"
        if report.read_bytes() != (files / report.name).read_bytes():
            raise ValueError(f"Review receipt differs from the archived record: {mode}")
        rows = json.loads(report.read_text())["clips"]
        if len(rows) != 6 or any(row["status"] != "ok" for row in rows):
            raise ValueError(f"Incomplete review outputs: {mode}")
        frames = count = 0
        observed = set()
        for row in rows:
            clip = row["clip_id"]
            if clip in observed:
                raise ValueError(f"Duplicate clip {clip}")
            observed.add(clip)
            path = inputs / Path(row["npz"]).name
            if digest(path) != row["sha256"]:
                raise ValueError(f"Unexpected prediction hash: {path}")
            with np.load(path, allow_pickle=False) as data:
                shared = {key: data[key] for key in SHARED}
                count += int(data["camera_valid"].sum())
                if not np.array_equal(
                    data["camera_valid"], shared["segment_index"] >= 0
                ):
                    raise ValueError(
                        f"Mesh presence does not match frozen support: {clip}"
                    )
                frames += len(shared["source_frames"])
                if mode == "gvhmr-archive":
                    axes[clip] = shared
                else:
                    for key in SHARED:
                        np.testing.assert_array_equal(
                            shared[key], axes[clip][key], err_msg=key
                        )
        mesh_count[mode] = count
    assert frames == 1711 and mesh_count == {"gvhmr-archive": 6617, "pt16": 6617}
    return {
        "run_head": RUN_HEAD,
        "trt_engine": "not_built",
        "export_paths_failed": 2,
        "clips": len(observed),
        "frame_rows": frames,
        "camera_mesh_rows": mesh_count,
        "review_prediction_files": 2 * len(observed),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("queue_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.queue_root, Path(__file__).resolve().parent)))
