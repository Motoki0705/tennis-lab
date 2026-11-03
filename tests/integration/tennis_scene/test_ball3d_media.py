import cv2
import numpy as np
from numpy.typing import NDArray

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
    plot_metrics_table,
    render_extremes,
    write_report,
)
from tests.support.ball3d_evaluation import make_annotations, make_inputs


def test_synchronized_two_method_video_full_decode_and_portable_reference_report(
    tmp_path,
):
    inputs = make_inputs()
    videos = {}
    for i, camera in enumerate(inputs.camera_ids):
        path = tmp_path / f"{camera}.avi"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter.fourcc(*"MJPG"),
            inputs.fps,
            inputs.source_size_wh,
        )
        assert writer.isOpened()
        for frame in range(inputs.num_frames):
            pixels: NDArray[np.uint8] = np.full((240, 320, 3), 30 + i * 40 + frame * 7, np.uint8)
            writer.write(pixels)
        writer.release()
        videos[camera] = path
    methods = (
        PipelineBaseline(min_frames=1),
        PipelineBaseline(min_frames=1, without_loess=True),
    )
    runs = {
        method.name: run_method(
            method,
            inputs,
            bundle_sha256="fixture",
            code_commit="fixture",
            code_fingerprint="fixture",
        )
        for method in methods
    }
    kinds: NDArray[np.str_] = np.full((3, 4), "observed", dtype="U22")
    kinds[1, 0], kinds[2, 2] = "estimated", "excluded_extrapolation"
    labels = make_annotations(inputs, np.tile([140.0, 170.0], (3, 4, 1)), kinds)
    with BallComparisonRenderer(
        inputs,
        {name: values[0] for name, values in runs.items()},
        videos,
        annotations=labels,
        width=640,
    ) as renderer:
        validation = render_video(renderer, tmp_path / "comparison.mp4")
    assert validation["decoded_frames"] == 4 and validation["fps"] == inputs.fps
    scores = score_comparison(runs, inputs, labels, config=ScoringConfig(split_frame=2))
    figure = tmp_path / "trajectories.png"
    plot_comparison(runs, inputs, figure, labels, split_frame=2)
    table = tmp_path / "metrics-table.png"
    plot_metrics_table(scores, table)
    crops = tmp_path / "extremes.png"
    assert (
        render_extremes(runs, inputs, videos, crops, labels, count=1)["status"]
        == "generated"
    )
    page = write_report(
        tmp_path,
        scores,
        video=tmp_path / "comparison.mp4",
        figure=figure,
        crops=crops,
        verification=validation,
        table_figure=table,
    )
    assert "approved estimates" in page.read_text()
    assert "Primary evaluation: [2,4)" in page.read_text()
    assert "data:video/mp4;base64," in page.read_text()
    assert broken_html_references(page) == []
    # Optional no-reference evaluation remains explicit and reports no accuracy.
    no_labels = score_comparison(
        runs, inputs, None, config=ScoringConfig(split_frame=2)
    )
    assert all(
        method["lovo_reprojection_px"]["all"] is None
        for method in no_labels["methods"].values()
    )
