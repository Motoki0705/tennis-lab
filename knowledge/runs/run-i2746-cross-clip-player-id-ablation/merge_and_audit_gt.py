"""Merge the 5 independent visual GT worker batches, audit consistency, and emit gt_cross_clip.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BENCH_ROOT = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4")
BATCH_DIR = Path("outputs/i2746_cross_clip_player_id/gt_batches")
OUT_GT = Path("knowledge/runs/run-i2746-cross-clip-player-id-ablation/gt_cross_clip.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    merged_clips: dict[str, dict] = {}
    batch_files = sorted(BATCH_DIR.glob("batch_*.json"))
    if len(batch_files) != 5:
        raise RuntimeError(f"Expected 5 batch files, found {len(batch_files)}: {[p.name for p in batch_files]}")
    for bf in batch_files:
        data = json.loads(bf.read_text())
        for cname, cdata in data["clips"].items():
            if cname in merged_clips:
                raise RuntimeError(f"Duplicate clip {cname} across batches")
            merged_clips[cname] = cdata

    expected_clips = [f"clip_{i:03d}" for i in range(69)]
    missing = [c for c in expected_clips if c not in merged_clips]
    if missing:
        raise RuntimeError(f"Missing clips in GT: {missing}")

    anomalies = []
    mixed_tracks = []
    uncertain_tracks = []
    end_change_clips = []
    prev_p01_side = None
    ref_p01_side = None

    clips_out = {}
    for ci, cname in enumerate(expected_clips):
        cdir = BENCH_ROOT / "videos/video_000/clips" / cname
        sj = json.loads((cdir / "scene/scene.json").read_text())
        msi_path = cdir / "scene" / sj["artifacts"]["multiview_scene_inputs"]["path"]
        msi = json.loads(msi_path.read_text())
        sides = [int(x) for x in msi["payload"]["people"]["sides"]]
        id_amb_arr = msi["payload"]["people"]["id_ambiguous"]["array"]
        import numpy as np

        id_amb = np.load(msi_path.parent / id_amb_arr)
        identities = np.load(msi_path.parent / msi["payload"]["people"]["grouped"]["identities"]["array"]).tolist()

        raw_entry = merged_clips[cname]
        players = sorted(raw_entry["players"], key=lambda x: int(x["local_player_id"]))
        gids = [p["gt_global_player_id"] for p in players]
        if sorted(g for g in gids if g is not None) != ["player_01", "player_02", "player_03", "player_04"]:
            anomalies.append({"clip": cname, "issue": "non_bijective", "gids": gids})

        gid_to_side = {p["gt_global_player_id"]: sides[int(p["local_player_id"])] for p in players if p["gt_global_player_id"]}
        if (
            gid_to_side.get("player_01") != gid_to_side.get("player_03")
            or gid_to_side.get("player_02") != gid_to_side.get("player_04")
            or gid_to_side.get("player_01") == gid_to_side.get("player_02")
        ):
            anomalies.append({"clip": cname, "issue": "doubles_side_mismatch", "gid_to_side": gid_to_side})

        p01_side = gid_to_side.get("player_01")
        if ref_p01_side is None:
            ref_p01_side = p01_side
        is_end_change_boundary = prev_p01_side is not None and p01_side != prev_p01_side
        if is_end_change_boundary:
            end_change_clips.append(cname)
        prev_p01_side = p01_side

        enriched_players = []
        for p in players:
            lid = int(p["local_player_id"])
            if p.get("purity") != "pure":
                mixed_tracks.append({"clip": cname, "local_player_id": lid, "purity": p.get("purity"), "mixed_interval": p.get("mixed_interval")})
            if p.get("status") != "confirmed" or p.get("gt_global_player_id") is None:
                uncertain_tracks.append({"clip": cname, "local_player_id": lid, "status": p.get("status")})
            enriched_players.append(
                {
                    "local_player_id": lid,
                    "scene_track_id": int(identities[lid]),
                    "side": sides[lid],
                    "gt_global_player_id": p["gt_global_player_id"],
                    "status": p["status"],
                    "purity": p["purity"],
                    "confidence": p.get("confidence", "high"),
                    "id_ambiguous_frames": int(id_amb[lid].sum()),
                    "mixed_interval": p.get("mixed_interval"),
                    "visual_evidence": p.get("visual_evidence", ""),
                }
            )

        clips_out[cname] = {
            "clip_idx": ci,
            "clip_id": f"video_000/{cname}",
            "end_flipped_vs_clip000": (p01_side != ref_p01_side),
            "is_end_change_boundary": is_end_change_boundary,
            "wardrobe_regime": "pre_layer_removal" if ci < 19 else "post_layer_removal",
            "players": enriched_players,
            "notes": raw_entry.get("notes", ""),
        }

    doc = {
        "schema_version": 1,
        "identity_scope": "meiji_4cam/video_000",
        "benchmark_version": "multiview-v4",
        "benchmark_manifest_sha256": sha256_file(BENCH_ROOT / "manifest.json"),
        "annotation_method": "independent_4view_plus_early_late_visual_inspection_by_gemini_4_hydrogen_xhigh_workers",
        "reid_model_independence": True,
        "player_definitions": {
            "player_01": {
                "team": "pair_13",
                "role": "net",
                "pre_layer_removal": "Red GOSEN long-sleeve jacket + black long pants (dark cap from clip_001)",
                "post_layer_removal": "Black short-sleeve t-shirt (white chest graphic) + black shorts + dark cap (from clip_019)",
            },
            "player_02": {
                "team": "pair_24",
                "role": "net",
                "pre_layer_removal": "Black long-sleeve jacket (large white 'G' logo on back) + black long pants, no cap",
                "post_layer_removal": "Dark/grey mottled short-sleeve t-shirt (white chest graphic) + black long pants, no cap (from clip_019)",
            },
            "player_03": {
                "team": "pair_13",
                "role": "baseline",
                "pre_layer_removal": "Black long-sleeve top + light-blue shorts + dark cap (cap removed by clip_005)",
                "post_layer_removal": "Black long-sleeve top + light-blue shorts, no cap",
            },
            "player_04": {
                "team": "pair_24",
                "role": "baseline",
                "pre_layer_removal": "White short-sleeve t-shirt + dark/black shorts + white cap",
                "post_layer_removal": "White short-sleeve t-shirt + dark/black shorts + white cap (unchanged)",
            },
        },
        "audit": {
            "num_clips": len(clips_out),
            "num_nodes": sum(len(v["players"]) for v in clips_out.values()),
            "anomalies": anomalies,
            "mixed_tracks": mixed_tracks,
            "uncertain_tracks": uncertain_tracks,
            "end_change_boundary_clips": end_change_clips,
            "num_end_flipped_clips": sum(1 for v in clips_out.values() if v["end_flipped_vs_clip000"]),
            "wardrobe_change_boundary_clip": "clip_019",
            "pre_layer_removal_clips": sum(1 for v in clips_out.values() if v["wardrobe_regime"] == "pre_layer_removal"),
            "post_layer_removal_clips": sum(1 for v in clips_out.values() if v["wardrobe_regime"] == "post_layer_removal"),
        },
        "clips": clips_out,
    }
    OUT_GT.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(json.dumps(doc["audit"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
