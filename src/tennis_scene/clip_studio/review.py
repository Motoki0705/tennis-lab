"""Browse an EXISTING Clip Studio project without edits, sync jobs or exports.

python -m src.tennis_scene.clip_studio.review --data-root /path/to/data \
    --source-directory tennis_multivew/raw/meiji_3cam/video_000 --port 8904
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import uvicorn
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.clip_studio.project import ClipStudioProject
from src.tennis_scene.clip_studio.web.app import create_app
from src.tennis_scene.configuration import (
    ClipStudioRuntimeConfig,
    parse_clip_studio_config,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.clip_studio_review",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "source_directory",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
    ),
)


def load_review_project(runtime: ClipStudioRuntimeConfig) -> ClipStudioProject:
    """Require a saved entry and exact raw source identity; never initialize it."""
    project = ClipStudioProject.load(
        runtime.export.projects_path,
        runtime.export.resolver,
        dataset_id=runtime.dataset_id,
        video_id=runtime.video_id,
    )
    saved = tuple((source.path, source.camera_id) for source in project.sources)
    discovered = tuple(zip(runtime.video_paths, runtime.camera_ids, strict=True))
    if saved != discovered:
        raise ValueError("保存projectのcamera/pathがraw動画と一致しません。")
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="既存data root（tennis_multivewを含む）",
    )
    parser.add_argument(
        "--source-directory", required=True, help="DATA相対のraw/<dataset>/<video>経路"
    )
    parser.add_argument("--port", type=int, default=8904)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        checkpoint_root=data_root,
        artifact_root=data_root,
        output_root=data_root,
        cache_root=data_root,
        external_asset_root=data_root,
    )
    resolver = PathResolver(roots)
    source = resolver.resolve(PathRole.DATA, args.source_directory)
    paths = PATH_BOUNDARY.validate(
        {"data_root": data_root, "source_directory": source},
        resolver=resolver,
    )
    cfg = OmegaConf.load(PROJECT_ROOT / "src/tennis_scene/configs/clip_studio.yaml")
    if not isinstance(cfg, DictConfig):
        raise TypeError("clip_studio config must be a mapping")
    del cfg["defaults"]
    del cfg["hydra"]
    cfg.paths = {key: str(value) for key, value in roots.as_mapping().items()}
    cfg.source_directory = (
        paths.declared("source_directory").path.relative_to(data_root).as_posix()
    )
    cfg.gui.port = args.port
    runtime = parse_clip_studio_config(cfg)
    project = load_review_project(runtime)
    cv2.setNumThreads(2)
    print(
        f"Clip Studio · 読取専用レビュー · http://127.0.0.1:{runtime.gui.port}",
        flush=True,
    )
    uvicorn.run(
        create_app(runtime, project, read_only=True),
        host="127.0.0.1",
        port=runtime.gui.port,
    )


if __name__ == "__main__":
    main()
