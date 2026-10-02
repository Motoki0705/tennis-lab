"""CPU-only asset and artifact-contract check for a fresh full-pipeline run.

Does not call ComponentRunner.run or load an inference model. The emitted CUDA
configuration records the adopted association settings for this run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    file_identity,
    standard_definition,
)
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.checksum import dual_sha256

CODE = Path(__file__).resolve().parents[2]


def compose_config(repo: Path, report: Path, clip: Path, device: str) -> DictConfig:
    manifest = ClipManifest.load(clip)
    media = [str(manifest.media_path(c).relative_to(repo / 'data')) for c in manifest.camera_ids]
    overrides = [
        f'paths.project_root={CODE}', f'paths.data_root={repo / "data"}',
        f'paths.checkpoint_root={repo / "ckpt"}', f'paths.external_asset_root={repo / "third_party"}',
        f'paths.output_root={report}', f'paths.artifact_root={report}',
        f'paths.cache_root={report / "cache"}', 'output_directory=run',
        f'video_paths={media}', f'camera_ids={list(manifest.camera_ids)}', f'device={device}',
        'people_models.runtime.vitpose.batch_size=4', 'people_models.runtime.hmr2.batch_size=4',
    ]
    with initialize_config_dir(version_base='1.3', config_dir=str(CODE / 'src/tennis_scene/configs')):
        config = compose(config_name='pipeline', overrides=overrides)
    return config


def check_full_recipe(runtime: PipelineRuntimeConfig) -> None:
    if not all(runtime.enabled.values()) or set(runtime.component_sources.values()) != {'execute'}:
        raise ValueError('Full qualification requires all features enabled and every component execute')
    if runtime.cache_source != 'execute':
        raise ValueError('Full qualification must start fresh')
    detector = runtime.people.runtime.dino_detector
    if runtime.people.detector != 'dino' or detector.confidence != .3 \
            or (detector.short_side, detector.max_long_side) != (800, 1333) \
            or runtime.tracking.method != 'strongsort_pp_pose' or runtime.merge_duplicate_person_boxes \
            or runtime.max_tracks_per_camera != 6:
        raise ValueError('The frozen detector/tracker/merge/selection contract changed')


def preflight(repo: Path, clip: Path, report: Path) -> dict[str, Any]:
    report.mkdir(parents=True, exist_ok=True)
    if (report / 'preflight.json').exists():
        raise FileExistsError('Preflight receipts are immutable; use a new report')
    config = compose_config(repo, report, clip, 'cpu')
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=True)
    check_full_recipe(runtime)
    manifest = ClipManifest.load(clip)
    source = build_clip_source(runtime.video_paths, runtime.camera_ids, clip_id=manifest.clip_id)
    if source.camera_ids != ('cam0', 'cam1', 'cam2'):
        raise ValueError('Qualification requires all three cameras')
    code = {str(p.relative_to(CODE)): dual_sha256(p) for p in sorted((CODE / 'src').rglob('*.py'))}
    nodes = standard_definition(runtime, source, code_identity='cpu-preflight-no-execution')
    runner = ComponentRunner(nodes, ClipStore(report / 'empty_store', json_value(source)))
    if runner.statuses or runner.references:
        raise AssertionError('Preflight must not execute components')
    assets = {name: file_identity(path) for name, path in enabled_model_assets(runtime).items()}
    # AFLink is small and loaded strictly on CPU to validate the published state dict.
    from src.tasks.person_tracking.strongsort_offline import AFLink
    AFLink(runtime.aflink_checkpoint)
    config.device = 'cuda'
    config_path = report / 'qualification.pipeline_config.yaml'
    config_path.write_text(OmegaConf.to_yaml(config, resolve=True))
    receipt = {
        'schema': 'full_pipeline_preflight_v1', 'status': 'ok', 'device_used': 'cpu',
        'inference_executed': False, 'source': json_value(source), 'assets': assets,
        'config': file_identity(config_path), 'code': code, 'order': list(runner.order),
        'nodes': {n.name: {'schema': n.io.output_schema, 'version': n.io.version,
                         'source': n.source, 'bindings': dict(n.bindings), 'settings': dict(n.settings)}
                  for n in nodes},
        'unverified': ['GPU inference and numerical artifact values', 'scene export and load-only restart',
                       'full-pipeline behavior of the selected association config', 'CUDA extension runtime'],
    }
    write_json_atomic(report / 'preflight.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.repo.resolve(), args.clip.resolve(), args.report.resolve())
    print(f'CPU preflight: {len(result["assets"])} assets, {len(result["nodes"])} nodes; no inference')


if __name__ == '__main__':
    main()
