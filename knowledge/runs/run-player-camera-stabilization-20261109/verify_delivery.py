"""Independently decode recovered comparison clips and check their source audit."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.tennis_scene.visualization.player_view.video import verify_video
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pinned = json.loads((args.inputs / "manifest.json").read_text())
    run = json.loads((args.run / "manifest.json").read_text())
    assert run["schema"] == "player_view_stabilization_comparison_run_v1"
    assert run["phase"] == "full"
    assert run["inputs_sha256"] == dual_sha256(args.inputs / "manifest.json")
    assert run["source_frames"] == list(range(240))
    assert run["human_acceptance"] == "pending"
    assert run["git_ref"] == pinned["git_ref"]
    expected = {"sigma_000": 0, "sigma_060": 0.06, "sigma_080": 0.08, "sigma_120": 0.12}
    assert {x["name"]: x["sigma_seconds"] for x in pinned["packages"]} == expected
    result = {
        "schema": "player_view_stabilization_delivery_verification_v1",
        "source_frames": [0, 240],
        "git_ref": run["git_ref"],
        "input_manifest_sha256": run["inputs_sha256"],
        "render_audit_rows_checked": 0,
        "videos": {},
        "human_acceptance": "pending",
    }
    for variant in pinned["packages"]:
        name = variant["name"]
        package = args.inputs / "packages" / name
        info = json.loads((package / "manifest.json").read_text())
        assert dual_sha256(package / "manifest.json") == variant["manifest_sha256"]
        videos = args.run / "videos" / name
        video = json.loads((videos / "manifest.json").read_text())
        assert run["variants"][name]["video_manifest_sha256"] == dual_sha256(
            videos / "manifest.json"
        )
        assert video["source_frames"] == list(range(240))
        assert video["player_ids"] == [0, 1]
        assert video["package_sha256"] == variant["manifest_sha256"]
        assert video["camera_processing"] == info["camera_processing"]
        assert video["display_lens"] == info["lens"]
        assert video["camera_processing"]["config"]["sigma_seconds"] == expected[name]
        assert video["human_acceptance"] == "pending"
        for player in (0, 1):
            path = args.run / "renders" / name / f"player_{player}" / "render.json"
            assert dual_sha256(path) == video["render_reports"][str(player)]
            audit = json.loads(path.read_text())
            assert audit["complete"] and audit["player_id"] == player
            assert audit["package_sha256"] == variant["manifest_sha256"]
            assert [x["source_frame"] for x in audit["frames"]] == list(range(240))
            assert all(
                "L4" in d["name"] and d["type"] == "OPTIX" for d in audit["gpu_devices"]
            )
            assert audit["camera"]["projection"] == "PERSP"
            assert audit["camera"]["resolution"] == [2560, 1440]
            assert audit["camera"]["depth_of_field"] is False
            assert audit["denoising"]["use_gpu"]
            assert video["camera_missing_frames"][str(player)] == [0, 1]
            viewer = next(x for x in info["players"] if x["player_id"] == player)
            with np.load(package / viewer["camera"]) as camera:
                for frame, row in enumerate(audit["frames"]):
                    assert row["cache_index"] == row["output_index"] == frame
                    assert row["camera_valid"] == bool(camera["valid"][frame])
                    assert row["camera_reason"] == int(camera["reason"][frame])
                    assert np.max(row["pc2_max_error_m"]) <= 1e-5
                    if row["camera_valid"]:
                        matrix = np.asarray(row["camera_matrix_world"])
                        np.testing.assert_allclose(
                            matrix[:3, :3], camera["rotation"][frame], atol=2e-5, rtol=0
                        )
                        np.testing.assert_allclose(
                            matrix[:3, 3], camera["location"][frame], atol=2e-5, rtol=0
                        )
                    else:
                        assert row["camera_matrix_world"] is None
            result["render_audit_rows_checked"] += len(audit["frames"])
        for filename, width in (
            ("player_0.mp4", 640),
            ("player_1.mp4", 640),
            ("comparison.mp4", 1280),
        ):
            path = videos / filename
            result["videos"][str(path.relative_to(args.run))] = verify_video(
                path, frames=240, fps=info["fps"], width=width, height=360
            )
    for player in (0, 1):
        path = args.run / "videos" / f"comparison_player_{player}.mp4"
        result["videos"][str(path.relative_to(args.run))] = verify_video(
            path, frames=240, fps=info["fps"], width=1280, height=720
        )
    result["decoded_video_frames"] = 240 * len(result["videos"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "videos": len(result["videos"]),
                "decoded_frames": result["decoded_video_frames"],
                "audit_rows": result["render_audit_rows_checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
