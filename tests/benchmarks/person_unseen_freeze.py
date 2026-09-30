"""Freeze the entire person recipe without reading reserved clip media or labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tasks.player_association.association.config import DEFAULT_CONFIG
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.definition import enabled_model_assets, file_identity
from src.utils.checksum import dual_sha256

CODE = Path(__file__).resolve().parents[2]
CLIPS = ('video_000/clip_002', 'video_001/clip_003', 'video_002/clip_003')
CAMERAS = ('cam0', 'cam1', 'cam2')


def configuration(repo: Path, report: Path) -> DictConfig:
    """Compose without binding/opening any input; batch four is the resource override."""
    with initialize_config_dir(version_base='1.3', config_dir=str(CODE / 'src/tennis_scene/configs')):
        cfg = compose(config_name='pipeline', overrides=[
            f'paths.project_root={CODE}', f'paths.data_root={repo / "data"}',
            f'paths.checkpoint_root={repo / "ckpt"}', f'paths.external_asset_root={repo / "third_party"}',
            f'paths.output_root={report}', f'paths.artifact_root={report}',
            f'paths.cache_root={report / "cache"}', 'output_directory=run', 'device=cuda',
            'people_models.runtime.vitpose.batch_size=4', 'people_models.runtime.hmr2.batch_size=4',
        ])
    return cfg


def source_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): dual_sha256(p) for p in sorted((root / 'src').rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.yaml', '.yml', '.json'}}


def external_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): dual_sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'}}


def freeze(repo: Path, report: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError('A freeze manifest is immutable')
    cfg = configuration(repo, report)
    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
    reservation = repo / 'data/tennis_multivew/processed/meiji_3cam/dataset/annotations/player_association/unseen_protocol.json'
    reserved = json.loads(reservation.read_text())  # metadata only, never a reserved clip
    if set(reserved['clips']) != set(CLIPS) or reserved['evaluation_attempts'] != 0:
        raise ValueError('Reservation differs or has already been evaluated')
    dino = repo / 'third_party/DINO'
    manifest = {
        'schema': 'i964_person_freeze_v1', 'created_at': datetime.now(UTC).isoformat(),
        'default_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
        'repo': str(repo), 'code_root': str(CODE), 'report': str(report),
        'unseen_clips': CLIPS, 'cameras': CAMERAS, 'reservation': file_identity(reservation),
        'unseen_media_opened': False, 'unseen_labels_opened': False, 'scoring_batches': 0,
        'pipeline_config': OmegaConf.to_container(cfg, resolve=True),
        'person': {
            'detector': runtime.people.detector, 'detector_runtime': json_value(runtime.people.runtime.dino_detector),
            'scope': 'full_frame', 'merge_duplicate_person_boxes': runtime.merge_duplicate_person_boxes,
            'merge_rule': 'greedy_iou_ge_0.8_score_desc_source_row_asc',
            'tracking': runtime.tracking.identity(),
            'pose_runtime': json_value(runtime.people.runtime.vitpose), 'pose_precision': 'float32',
            'pose_confidence': 'raw finite heatmap maxima; no sigmoid/clipping',
            'clip': {'name': runtime.tracking.encoder, 'input_size_hw': [256, 128],
                     'rgb_range': [0, 1], 'mean': [.5, .5, .5], 'std': [.5, .5, .5],
                     'feature': 'L2-normalized CLS + projection (1280)'},
            'aflink': {'input_length': 30, 'gap_frames_exclusive': [0, 30], 'max_endpoint_distance_px': 75,
                       'max_cost_exclusive': .05, 'normalization_epsilon': 1e-5,
                       'coordinate': 'source pixel bbox top-left', 'assignment': 'Hungarian',
                       'source_ids_preserved': True},
            'gsi': {'gap_frames_exclusive': [1, 20], 'length_scale': 'clip(10*log(1000/N), .1, 100)',
                    'kernel_noise': 1e-10, 'synthetic_separate_from_observed': True},
            'selection': asdict(LinkingConfig(max_candidates=runtime.max_tracks_per_camera)),
            'selection_region': {'core_half_width_m': 4.115, 'corridor_half_width_m': 5.485,
                                 'half_length_plus_runoff_m': 16.885, 'endpoint_window_s': .1,
                                 'switch_window_s': .25, 'switch_max_jump_m': 3.,
                                 'cap_after_dwell_and_linking': True, 'keep_all_selected_real_observations': True},
            'appearance_sampling': asdict(CropSamplingConfig()),
            'association': asdict(runtime.player_association), 'association_config': file_identity(DEFAULT_CONFIG),
            'v3_safety': 'switch segmentation / ambiguous short-fragment exclusion / handoff / framewise IDs; no fill',
            'person_observations': json_value(cfg.person_observations),
        },
        # Body/court assets are included even though the unseen job stops before 3D reconstruction.
        'assets': {k: file_identity(p) for k, p in enabled_model_assets(runtime).items()},
        'source': source_hashes(CODE),
        'dino_source': {'root': str(dino), 'commit': subprocess.check_output(
            ['git', '-C', str(dino), 'rev-parse', 'HEAD'], text=True).strip(), 'files': external_hashes(dino)},
        'environment_files': [file_identity(CODE / name) for name in ('uv.lock', 'pyproject.toml')],
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'torchvision', 'numpy', 'scipy', 'timm', 'opencv-python', 'omegaconf', 'hydra-core')},
        'generator': file_identity(Path(__file__)),
        'evaluation': {
            'attempts_allowed': 1, 'retune_after_opening': False, 'fit_or_dev_rerun': False,
            'camera_sides': 'same as dev: i932 decisions_v2.json annotation-ball view_half_turns; no court_side execution',
            'near_far': 'same dev tracking_units convention: rank each labelled player bbox bottom within camera/frame',
            'metrics': ['raw/group IDF1', 'camera-local switches/fragments', '#933 pair F1', 'players kept',
                        'non-players left', 'camera x near/far', 'coverage/unknown', 'undecided reasons'],
            'videos': 'one full-length synchronized 3-camera video per clip; prediction identity / raw IDs / selection',
            'labels': 'Create/review unseen box labels only after this freeze, without changing predictions; score once next run.',
            'qualification': 'clip_000 full pipeline remains pending; ball ft-e13/court_side unchanged',
        },
        'budget': {'resource': 'all', 'wall_seconds': 7170, 'allocator_bytes': 7 * 1024**3,
                   'vram_stop_bytes': 9_000_000_000, 'disk_limit_bytes': 4_800_000_000,
                   'outer_timeout_seconds': 7190, 'kill_after_seconds': 10,
                   'cpu_threads': 4, 'build_jobs': 2, 'minimum_available_ram_bytes': 6 * 1024**3},
    }
    write_json_atomic(target, manifest)


def require_pushed(path: Path, commit: str, *, root: Path = CODE,
                   remote: str = 'origin/campaign930/i964-2-tracking') -> dict[str, Any]:
    """Gate before *any* reserved clip read; exact manifest must be committed and pushed."""
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=root, check=True)
    blob = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(root)}'], cwd=root)
    if hashlib.sha256(blob).hexdigest() != dual_sha256(path):
        raise ValueError('Freeze manifest changed since the pushed commit')
    manifest: dict[str, Any] = json.loads(blob)
    if manifest['schema'] != 'i964_person_freeze_v1' or manifest['unseen_media_opened'] \
            or manifest['unseen_labels_opened'] or manifest['scoring_batches'] != 0:
        raise ValueError('Require the pre-opening freeze')
    return manifest


def verify(manifest: dict[str, Any]) -> None:
    """Fail closed on any code, config, checkpoint or runtime dependency drift."""
    if source_hashes(CODE) != manifest['source']:
        raise ValueError('Frozen source/config changed')
    for record in (*manifest['assets'].values(), *manifest['environment_files'],
                   manifest['person']['association_config'], manifest['reservation'], manifest['generator']):
        if file_identity(Path(record['path'])) != record:
            raise ValueError(f'Frozen input changed: {record["path"]}')
    external = manifest['dino_source']
    if external_hashes(Path(external['root'])) != external['files']:
        raise ValueError('Frozen DINO source changed')
    for name, version in manifest['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Frozen package changed: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'report', 'target'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.repo.resolve(), args.report.resolve(), args.target.resolve())
