import json

from omegaconf import OmegaConf

from src.wasb.scripts.generate_dataset import META_FILENAME, reset_videos


def test_reset_failed_only_resets_failed(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    meta = {
        "version": "1.0",
        "created_at": "t0",
        "updated_at": "t0",
        "next_game_id": 11,
        "videos": {
            "a.mp4": {
                "status": "failed",
                "output_game": "game11",
                "num_clips": None,
                "processed_at": None,
                "file_hash": "sha256:deadbeef",
                "error_message": "boom",
            },
            "b.mp4": {
                "status": "completed",
                "output_game": "game12",
                "num_clips": 3,
                "processed_at": "t1",
                "file_hash": "sha256:cafebabe",
                "error_message": None,
            },
        },
    }
    (output_dir / META_FILENAME).write_text(json.dumps(meta))

    cfg = OmegaConf.create(
        {"mode": "reset_failed", "output_dir": str(output_dir), "reset_video": []}
    )
    assert reset_videos(cfg) == 0

    updated = json.loads((output_dir / META_FILENAME).read_text())
    assert updated["videos"]["a.mp4"]["status"] == "pending"
    assert updated["videos"]["a.mp4"]["error_message"] is None
    assert updated["videos"]["b.mp4"]["status"] == "completed"


def test_reset_video_resets_only_named_videos(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    meta = {
        "version": "1.0",
        "created_at": "t0",
        "updated_at": "t0",
        "next_game_id": 11,
        "videos": {
            "a.mp4": {
                "status": "completed",
                "output_game": "game11",
                "num_clips": 1,
                "processed_at": "t1",
                "file_hash": "sha256:deadbeef",
                "error_message": None,
            },
            "b.mp4": {
                "status": "failed",
                "output_game": "game12",
                "num_clips": None,
                "processed_at": None,
                "file_hash": "sha256:cafebabe",
                "error_message": "boom",
            },
        },
    }
    (output_dir / META_FILENAME).write_text(json.dumps(meta))

    cfg = OmegaConf.create(
        {
            "mode": "reset_video",
            "output_dir": str(output_dir),
            "reset_video": ["a.mp4"],
        }
    )
    assert reset_videos(cfg) == 0

    updated = json.loads((output_dir / META_FILENAME).read_text())
    assert updated["videos"]["a.mp4"]["status"] == "pending"
    assert updated["videos"]["b.mp4"]["status"] == "failed"

