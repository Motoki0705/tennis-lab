"""FastAPI app for the video annotation tool.

This backend is intentionally minimal and designed for local usage with a
single video under ``data/tmp``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.tools.annotation.backend.exporters import (
    CourtExportConfig,
    WasbExportConfig,
    export_court_keypoints,
    export_wasb_clip,
)
from src.tools.annotation.backend.models import (
    BallClipConfig,
    BallFrameAnnotation,
    CourtFrameAnnotation,
    ExportResult,
    VideoMeta,
)
from src.tools.annotation.backend.state import AnnotationState, StatePaths
from src.tools.annotation.backend.video import VideoFrameProvider
from src.utils.geometry.constants import COURT_KP_NAMES, NUM_COURT_KP


def create_app(video_path: str | Path, output_root: str | Path) -> FastAPI:
    """Create a FastAPI app instance."""
    provider = VideoFrameProvider(video_path)
    state = AnnotationState(StatePaths(Path(output_root)))

    app = FastAPI(title="tennis-lab annotation backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/meta", response_model=VideoMeta)
    def get_meta() -> VideoMeta:
        info = provider.info
        return VideoMeta(
            fps=info.fps,
            frame_count=info.frame_count,
            width=info.width,
            height=info.height,
        )

    @app.get("/api/frame/{frame_idx}.jpg")
    def get_frame(frame_idx: int) -> Response:
        try:
            data = provider.encode_jpeg(frame_idx)
        except IndexError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:  # pragma: no cover - runtime decode errors
            raise HTTPException(status_code=500, detail=str(e)) from e
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/ball/clip_config", response_model=BallClipConfig)
    def get_ball_clip_config() -> BallClipConfig:
        return state.load_ball_clip_config()

    @app.put("/api/ball/clip_config", response_model=BallClipConfig)
    def put_ball_clip_config(cfg: BallClipConfig) -> BallClipConfig:
        meta = provider.info
        if cfg.start_frame >= meta.frame_count:
            raise HTTPException(status_code=400, detail="start_frame out of range")
        if cfg.start_frame + cfg.clip_length > meta.frame_count:
            raise HTTPException(status_code=400, detail="clip exceeds video length")
        state.save_ball_clip_config(cfg)
        return cfg

    @app.get("/api/ball/annotations/{local_idx}", response_model=BallFrameAnnotation)
    def get_ball_annotation(local_idx: int) -> BallFrameAnnotation:
        cfg = state.load_ball_clip_config()
        if local_idx < 0 or local_idx >= cfg.clip_length:
            raise HTTPException(status_code=400, detail="local_idx out of clip range")
        return state.load_ball_annotation(local_idx)

    @app.put("/api/ball/annotations/{local_idx}", response_model=BallFrameAnnotation)
    def put_ball_annotation(local_idx: int, ann: BallFrameAnnotation) -> BallFrameAnnotation:
        cfg = state.load_ball_clip_config()
        if local_idx < 0 or local_idx >= cfg.clip_length:
            raise HTTPException(status_code=400, detail="local_idx out of clip range")
        state.save_ball_annotation(local_idx, ann)
        return ann

    @app.get("/api/court/kp_names")
    def get_court_kp_names() -> list[str]:
        return list(COURT_KP_NAMES)

    @app.get("/api/court/annotations/{frame_idx}", response_model=CourtFrameAnnotation)
    def get_court_annotation(frame_idx: int) -> CourtFrameAnnotation:
        return state.load_court_annotation(frame_idx, NUM_COURT_KP)

    @app.put("/api/court/annotations/{frame_idx}", response_model=CourtFrameAnnotation)
    def put_court_annotation(frame_idx: int, ann: CourtFrameAnnotation) -> CourtFrameAnnotation:
        if ann.frame_idx != frame_idx:
            ann.frame_idx = frame_idx
        if len(ann.keypoints) != NUM_COURT_KP:
            raise HTTPException(
                status_code=400,
                detail=f"keypoints must have length {NUM_COURT_KP}",
            )
        state.save_court_annotation(ann)
        return ann

    @app.get("/api/court/annotated_frames")
    def list_court_frames() -> list[int]:
        return state.list_court_annotated_frames()

    @app.post("/api/export/wasb", response_model=ExportResult)
    def export_ball() -> ExportResult:
        cfg = state.load_ball_clip_config()
        meta = provider.info
        if cfg.start_frame + cfg.clip_length > meta.frame_count:
            raise HTTPException(status_code=400, detail="clip exceeds video length")

        annotations: dict[int, tuple[float, float, int, float]] = {}
        for local_idx in range(cfg.clip_length):
            ann = state.load_ball_annotation(local_idx)
            annotations[local_idx] = (ann.x_px, ann.y_px, int(ann.visibility), ann.score)

        out_dir = export_wasb_clip(
            provider=provider,
            out_cfg=WasbExportConfig(output_dir=Path(output_root)),
            start_frame=cfg.start_frame,
            clip_length=cfg.clip_length,
            annotations_by_local=annotations,
        )
        return ExportResult(output_dir=str(out_dir))

    @app.post("/api/export/court", response_model=ExportResult)
    def export_court() -> ExportResult:
        frame_indices = state.list_court_annotated_frames()
        anns = [state.load_court_annotation(idx, NUM_COURT_KP) for idx in frame_indices]
        out_dir = export_court_keypoints(
            provider=provider,
            out_cfg=CourtExportConfig(output_dir=Path(output_root)),
            annotations=anns,
        )
        return ExportResult(output_dir=str(out_dir))

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotation backend server")
    parser.add_argument(
        "--video",
        type=str,
        default="data/tmp/input.mp4",
        help="Path to input video (default: data/tmp/input.mp4)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/tmp",
        help="Output root directory (default: data/tmp)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the annotation backend server."""
    args = _parse_args()
    import uvicorn

    app = create_app(args.video, args.out)
    uvicorn.run(app, host=args.host, port=args.port, reload=bool(args.reload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

