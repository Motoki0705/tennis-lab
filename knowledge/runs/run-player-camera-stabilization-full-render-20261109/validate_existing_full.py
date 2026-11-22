"""Strict read-only validation of the original full240 artifacts; no rendering."""

import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

BASE = Path("outputs/tennis_scene/visualize/player_view/stabilization-comparison")
OPS = BASE / "operations/attempt2"
REPAIR = OPS / "resume1"
FULL = BASE / "attempt2/full"
SCRATCH = BASE / "attempt2/full-render"
REF = "d133993421c206af28340f4ffe082c65b70fe7a7"
sys.path.insert(0, str(Path.cwd()))
from src.tennis_scene.visualization.player_view.video import validate_render


def read(path):
    return json.loads(path.read_text())


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    active = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            argv = process.joinpath('cmdline').read_bytes().decode(errors='replace').split(chr(0))[:-1]
            if argv and (Path(argv[0]).name == 'blender' or any(a.endswith('/render_full_comparison.py') for a in argv)):
                active.append({'pid': int(process.name), 'argv': argv})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    assert not active, active
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == REF
    helpers = read(OPS / "full-driver-source-digests.json")
    for name, expected in helpers.items():
        assert digest(OPS / name) == expected
    inputs = read(BASE / "inputs/manifest.json")
    assert digest(BASE / "inputs/manifest.json") == read(OPS / "gpu-operation.json")["input_manifest_sha256"]
    assert inputs["git_ref"] == REF and inputs["source_frame_interval"] == [0, 240] and inputs["player_ids"] == [0, 1]
    assert [v["sigma_seconds"] for v in inputs["packages"]] == [0.0, 0.06, 0.08, 0.12]
    summary = read(FULL / "manifest.json")
    assert summary["phase"] == "full" and summary["git_ref"] == REF
    assert summary["source_frames"] == list(range(240)) and summary["archived_source_pngs"] == 1920
    assert summary["driver_sha256"] == helpers["render_full_comparison.py"] and summary["archive_helper_sha256"] == helpers["archive_render_frames.py"]
    assert summary["inputs_sha256"] == digest(BASE / "inputs/manifest.json")
    views = []
    preserved = []
    for variant in inputs["packages"]:
        name = variant["name"]
        package = BASE / "inputs/packages" / name
        info = read(package / "manifest.json")
        assert digest(package / "manifest.json") == variant["manifest_sha256"]
        assert info["camera_processing"]["config"]["sigma_seconds"] == variant["sigma_seconds"]
        for filename, expected in info["files"].items():
            assert digest(package / filename) == expected
        published = read(FULL / "videos" / name / "manifest.json")
        assert published["source_frames"] == list(range(240)) and published["preview"] is False and published["player_ids"] == [0, 1]
        assert published["display_lens"] == info["lens"]
        assert summary["variants"][name]["sigma_seconds"] == variant["sigma_seconds"] and summary["variants"][name]["lens"] == info["lens"]
        assert summary["variants"][name]["video_manifest_sha256"] == digest(FULL / "videos" / name / "manifest.json")
        for player in [0, 1]:
            render = SCRATCH / "renders" / name / f"player_{player}"
            report = validate_render(package, render, player)
            assert [r["source_frame"] for r in report["frames"]] == list(range(240))
            assert report["blender_version"] == "5.2.1 LTS"
            assert report["gpu_devices"] and all(d["type"] == "OPTIX" and "L4" in d["name"] for d in report["gpu_devices"])
            assert report["denoising"] == {"enabled": True, "denoiser": "OPENIMAGEDENOISE", "use_gpu": True}
            audit = FULL / "renders" / name / f"player_{player}" / "render.json"
            assert digest(audit) == digest(render / "render.json") == published["render_reports"][str(player)]
            archive_path = FULL / "frame_archives" / name / f"player_{player}.zip"
            receipt = read(archive_path.with_suffix(".manifest.json"))
            assert receipt["members"] == 240 and receipt["source_frames"] == list(range(240)) and receipt["all_uncompressed_png_sha256_verified"] is True
            assert digest(archive_path) == receipt["sha256"] and digest(archive_path, "md5") == receipt["md5"] and archive_path.stat().st_size == receipt["bytes"]
            assert receipt["render_report_sha256"] == digest(audit)
            assert summary["raw_frame_archives"][name][str(player)]["sha256"] == receipt["sha256"]
            with zipfile.ZipFile(archive_path) as saved:
                payload = saved.read("manifest.json")
                assert hashlib.sha256(payload).hexdigest() == receipt["manifest_sha256"]
                original = json.loads(payload)
                assert original["player_id"] == player and original["source_frames"] == list(range(240))
                assert original["render_report_sha256"] == digest(audit) and original["resolution"] == report["camera"]["resolution"]
                assert saved.namelist() == [f"frames/{i:05d}.png" for i in range(240)] + ["manifest.json"]
                for index, row in enumerate(original["files"]):
                    png = render / row["path"]
                    assert row["source_frame"] == index and row["camera_valid"] == report["frames"][index]["camera_valid"]
                    assert png.stat().st_size == row["bytes"] and digest(png) == row["sha256"]
                    with png.open("rb") as stream:
                        header = stream.read(24)
                    assert header[:8] == b"\x89PNG\r\n\x1a\n" and list(struct.unpack(">II", header[16:24])) == report["camera"]["resolution"]
                    with saved.open(row["path"]) as stream:
                        assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
                    assert saved.getinfo(row["path"]).file_size == row["bytes"]
            views.append({"variant": name, "player_id": player, "frames": 240, "original_png_hashes_and_zip_bytes_verified": True, "archive_sha256": receipt["sha256"], "render_report_sha256": digest(audit)})
        marker = read(FULL / "checkpoints" / f"{name}.json")
        assert marker["source_frames"] == list(range(240)) and marker["sigma_seconds"] == variant["sigma_seconds"]
        for member in marker["files"]:
            path = FULL / member["path"]
            assert path.stat().st_size == member["bytes"] and digest(path) == member["sha256"] and digest(path, "md5") == member["md5"]
    mp4s = list((FULL / "videos").glob("**/*.mp4"))
    assert len(mp4s) == 14 and all(p.stat().st_size > 0 for p in mp4s)
    assert set(summary["comparison_videos"]) == {"0", "1"}
    for path in sorted(p for p in FULL.rglob("*") if p.is_file()):
        assert not path.is_symlink() and not path.name.endswith('.partial'), path
        preserved.append({"path": path.relative_to(FULL).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path), "md5": digest(path, "md5")})
    result = {"schema_version": 1, "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "source_ref": REF, "helper_sha256": helpers, "views": views, "source_pngs": 1920, "archives": 8, "mp4s": 14, "audits": 8, "frames_per_view": 240, "rendered_or_modified_images": 0, "full_cpu_decode_or_pts_repeated": False, "files": preserved, "human_acceptance": "pending"}
    destination = REPAIR / "persist/validated-full.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    print("EXISTING_FULL_VALIDATED " + json.dumps({"views": 8, "frames_per_view": 240, "source_pngs": 1920, "archives": 8, "mp4s": 14, "audits": 8, "rerendered_views": 0}), flush=True)


if __name__ == "__main__":
    main()
