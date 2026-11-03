import cv2
import numpy as np

from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.evaluation.ball3d.runner import run_method
from src.tennis_scene.evaluation.ball3d.scoring import ScoringConfig, score_comparison
from src.tennis_scene.visualization.ball3d.rendering import (
    BallComparisonRenderer,
    render_video,
)
from src.tennis_scene.visualization.ball3d.report import (
    broken_html_references,
    plot_comparison,
    write_report,
)
from tests.support.ball3d_evaluation import make_inputs


def test_synchronized_video_full_decode_and_portable_no_gt_report(tmp_path):
    inputs = make_inputs()
    videos = {}
    for i, camera in enumerate(inputs.camera_ids):
        path = tmp_path / f"{camera}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), inputs.fps, inputs.source_size_wh)
        assert writer.isOpened()
        for frame in range(inputs.num_frames):
            pixels = np.full((240, 320, 3), 30 + i * 40 + frame * 7, np.uint8)
            writer.write(pixels)
        writer.release()
        videos[camera] = path
    method = PipelineBaseline(min_frames=1)
    runs = run_method(method, inputs, bundle_sha256="fixture", code_commit="fixture", code_fingerprint="fixture")
    with BallComparisonRenderer(inputs, {method.name: runs[0]}, videos, width=640) as renderer:
        validation = render_video(renderer, tmp_path / "comparison.mp4")
    assert validation["decoded_frames"] == 4 and validation["fps"] == inputs.fps
    scores = score_comparison({method.name: runs}, inputs, None, config=ScoringConfig(split_frame=2))
    figure = tmp_path / "trajectories.png"
    plot_comparison({method.name: runs}, inputs, figure, split_frame=2)
    page = write_report(tmp_path, scores, video=tmp_path / "comparison.mp4", figure=figure,
                         crops=None, verification=validation)
    assert "GT unavailable" in page.read_text()
    assert "Primary evaluation: [2,4)" in page.read_text()
    assert "data:video/mp4;base64," in page.read_text()
    assert broken_html_references(page) == []
