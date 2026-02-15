"""FastAPI app for the video annotation tool.

This backend is intentionally minimal and designed for local usage with a
single video under ``data/tmp``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.tools.annotation.backend.ball_assist import (
    BallAssistConfig,
    build_assist_meta,
    run_ball_assist_for_clip,
)
from src.tools.annotation.backend.exporters import (
    CourtExportConfig,
    WasbExportConfig,
    export_court_keypoints,
    export_wasb_clip,
)
from src.tools.annotation.backend.homography import (
    fill_court_keypoints_from_homography,
)
from src.tools.annotation.backend.models import (
    BallAssistAll,
    BallAssistRunRequest,
    BallAssistRunResult,
    BallAssistState,
    BallAssistSummary,
    BallClipConfig,
    BallFrameAnnotation,
    CourtFrameAnnotation,
    ExportResult,
    VideoMeta,
)
from src.tools.annotation.backend.state import AnnotationState, StatePaths
from src.tools.annotation.backend.video import VideoFrameProvider
from src.utils.schema.court import COURT_KP_NAMES, NUM_COURT_KP


def create_app(
    video_path: str | Path,
    output_root: str | Path,
    assist_cfg: BallAssistConfig | None = None,
) -> FastAPI:
    """Create a FastAPI app instance."""
    provider = VideoFrameProvider(video_path)
    state = AnnotationState(StatePaths(Path(output_root)))
    if assist_cfg is None:
        assist_cfg = BallAssistConfig(
            checkpoint_path=None,
            model_type="wasb",
            device="cpu",
            batch_size=64,
            score_threshold=0.5,
            max_disp=300,
        )

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

    @app.delete("/api/ball/annotations/{local_idx}")
    def delete_ball_annotation(local_idx: int) -> dict[str, bool]:
        cfg = state.load_ball_clip_config()
        if local_idx < 0 or local_idx >= cfg.clip_length:
            raise HTTPException(status_code=400, detail="local_idx out of clip range")
        state.delete_ball_annotation(local_idx)
        return {"ok": True}

    @app.get("/api/ball/annotated_frames")
    def list_ball_frames() -> list[int]:
        return state.list_ball_annotated_frames()

    @app.get("/api/ball/assist/summary", response_model=BallAssistSummary)
    def get_ball_assist_summary() -> BallAssistSummary:
        assist_state = state.load_ball_assist_state()
        if assist_state is None:
            return BallAssistSummary(
                available=False,
                clip_matches_current=False,
                clip=None,
                meta=None,
                count=0,
            )
        clip_cfg = state.load_ball_clip_config()
        matches = (
            assist_state.clip.start_frame == clip_cfg.start_frame
            and assist_state.clip.clip_length == clip_cfg.clip_length
        )
        return BallAssistSummary(
            available=True,
            clip_matches_current=matches,
            clip=assist_state.clip,
            meta=assist_state.meta,
            count=len(assist_state.annotations),
        )

    @app.get("/api/ball/assist/all", response_model=BallAssistAll)
    def get_ball_assist_all() -> BallAssistAll:
        assist_state = state.load_ball_assist_state()
        if assist_state is None:
            raise HTTPException(status_code=404, detail="assist not available")
        clip_cfg = state.load_ball_clip_config()
        if (
            assist_state.clip.start_frame != clip_cfg.start_frame
            or assist_state.clip.clip_length != clip_cfg.clip_length
        ):
            raise HTTPException(status_code=409, detail="assist clip mismatch")
        return BallAssistAll(annotations=assist_state.annotations)

    @app.get("/api/ball/assist/{local_idx}", response_model=BallFrameAnnotation)
    def get_ball_assist_frame(local_idx: int) -> BallFrameAnnotation:
        assist_state = state.load_ball_assist_state()
        if assist_state is None:
            raise HTTPException(status_code=404, detail="assist not available")
        clip_cfg = state.load_ball_clip_config()
        if (
            assist_state.clip.start_frame != clip_cfg.start_frame
            or assist_state.clip.clip_length != clip_cfg.clip_length
        ):
            raise HTTPException(status_code=409, detail="assist clip mismatch")
        ann = assist_state.annotations.get(local_idx)
        if ann is None:
            raise HTTPException(status_code=404, detail="assist frame not found")
        return ann

    @app.post("/api/ball/assist/run", response_model=BallAssistRunResult)
    def run_ball_assist(req: BallAssistRunRequest | None = None) -> BallAssistRunResult:
        cfg = assist_cfg
        if req is None:
            req = BallAssistRunRequest()

        merged = BallAssistConfig(
            checkpoint_path=(
                Path(req.checkpoint_path)
                if req.checkpoint_path is not None
                else cfg.checkpoint_path
            ),
            model_type=req.model_type or cfg.model_type,
            device=req.device or cfg.device,
            batch_size=req.batch_size or cfg.batch_size,
            score_threshold=(
                req.score_threshold if req.score_threshold is not None else cfg.score_threshold
            ),
            max_disp=req.max_disp or cfg.max_disp,
        )

        clip_cfg = state.load_ball_clip_config()
        meta = provider.info
        if clip_cfg.start_frame + clip_cfg.clip_length > meta.frame_count:
            raise HTTPException(status_code=400, detail="clip exceeds video length")

        try:
            annotations = run_ball_assist_for_clip(
                provider=provider,
                clip_cfg=clip_cfg,
                assist_cfg=merged,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # pragma: no cover - runtime inference errors
            raise HTTPException(status_code=500, detail=str(e)) from e

        assist_state = BallAssistState(
            clip=clip_cfg,
            meta=build_assist_meta(merged),
            annotations=annotations,
        )
        state.save_ball_assist_state(assist_state)
        return BallAssistRunResult(
            clip=clip_cfg,
            meta=assist_state.meta,
            count=len(annotations),
        )

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

    @app.delete("/api/court/annotations/{frame_idx}")
    def delete_court_annotation(frame_idx: int) -> dict[str, bool]:
        if frame_idx < 0:
            raise HTTPException(status_code=400, detail="frame_idx out of range")
        state.delete_court_annotation(frame_idx)
        return {"ok": True}

    @app.post("/api/court/homography", response_model=CourtFrameAnnotation)
    def run_court_homography(ann: CourtFrameAnnotation) -> CourtFrameAnnotation:
        if len(ann.keypoints) != NUM_COURT_KP:
            raise HTTPException(
                status_code=400,
                detail=f"keypoints must have length {NUM_COURT_KP}",
            )
        try:
            filled = fill_court_keypoints_from_homography(ann)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        state.save_court_annotation(filled)
        return filled

    @app.get("/api/court/annotated_frames")
    def list_court_frames() -> list[int]:
        return state.list_court_annotated_frames()

    @app.post("/api/export/wasb", response_model=ExportResult)
    def export_ball() -> ExportResult:
        cfg = state.load_ball_clip_config()
        meta = provider.info
        if cfg.start_frame + cfg.clip_length > meta.frame_count:
            raise HTTPException(status_code=400, detail="clip exceeds video length")

        assist_state = state.load_ball_assist_state()
        assist_ok = False
        if assist_state is not None:
            assist_ok = (
                assist_state.clip.start_frame == cfg.start_frame
                and assist_state.clip.clip_length == cfg.clip_length
            )

        annotations: dict[int, tuple[float, float, int, float]] = {}
        for local_idx in range(cfg.clip_length):
            if state.has_ball_annotation(local_idx):
                ann = state.load_ball_annotation(local_idx)
                annotations[local_idx] = (
                    ann.x_px,
                    ann.y_px,
                    int(ann.visibility),
                    ann.score,
                )
                continue
            if assist_ok and assist_state is not None:
                assist = assist_state.annotations.get(local_idx)
                if assist is not None:
                    annotations[local_idx] = (
                        assist.x_px,
                        assist.y_px,
                        int(assist.visibility),
                        assist.score,
                    )
                    continue
            annotations[local_idx] = (0.0, 0.0, 0, 0.0)

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
        default="data/tennis/raw/videos/10 Minutes Of Kei Nishikori MAGIC.mp4",
        help="Path to input video (default: data/tmp/input.mp4)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/tennis/game11",
        help="Output root directory (default: data/tmp)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--assist-checkpoint",
        type=str,
        default=None,
        help="Path to WASB checkpoint for assist mode",
    )
    parser.add_argument(
        "--assist-model",
        type=str,
        choices=["wasb", "hrcnet"],
        default="wasb",
        help="Model type for assist mode",
    )
    parser.add_argument(
        "--assist-device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for assist inference",
    )
    parser.add_argument(
        "--assist-batch-size",
        type=int,
        default=64,
        help="Batch size for assist inference",
    )
    parser.add_argument(
        "--assist-score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for assist inference",
    )
    parser.add_argument(
        "--assist-max-disp",
        type=int,
        default=300,
        help="Max tracker displacement for assist inference",
    )
    return parser.parse_args()


def main() -> int:
    """Run the annotation backend server."""
    args = _parse_args()
    import uvicorn

    assist_cfg = BallAssistConfig(
        checkpoint_path=Path(args.assist_checkpoint)
        if args.assist_checkpoint
        else None,
        model_type=args.assist_model,
        device=args.assist_device,
        batch_size=args.assist_batch_size,
        score_threshold=args.assist_score_threshold,
        max_disp=args.assist_max_disp,
    )
    app = create_app(args.video, args.out, assist_cfg=assist_cfg)
    uvicorn.run(app, host=args.host, port=args.port, reload=bool(args.reload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

