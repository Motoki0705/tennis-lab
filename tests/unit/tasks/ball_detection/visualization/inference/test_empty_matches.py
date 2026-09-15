from src.tasks.ball_detection.visualization.inference.service import DetectionService


def test_no_match_distance_is_undefined(
    tmp_path, make_clip_dataset, make_tiny_checkpoint
):
    make_clip_dataset(tmp_path / "data" / "tennis" / "tracknet")
    make_tiny_checkpoint(
        tmp_path / "outputs" / "ball_detection" / "tiny.ckpt", num_frames=2
    )
    result = DetectionService(tmp_path).infer(
        "tiny.ckpt", "tracknet::game1/Clip1", count=2, threshold=1.0, device="cpu"
    )
    assert result["metrics"]["matched_detections"] == 0
    assert result["metrics"]["mean_distance_px"] is None
