"""Optional cache migration for v7; a fresh one-command build computes these itself."""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
import numpy as np
from src.tennis_scene.dataset_pipeline.build import materialize_dataset
from src.tennis_scene.dataset_pipeline.features import validated_feature_cache
from src.tennis_scene.generate_dataset.manifest import ClipManifest, load_dataset_manifest
from src.tasks.slcs.data.dino_tokens import dino_dir, load_dino_spec
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.io import save_json_atomic

root = Path.cwd()
source = root / 'data/tennis_multivew/processed/meiji_3cam/dataset'
old_dataset = root / 'data/slcs/meiji_rgb_v6'
new_dataset = root / 'data/slcs/meiji_rgb_v7'
old_observations = root / 'outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-003'
new_observations = root / 'outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004'
manifest = load_dataset_manifest(source)
selected = tuple(sorted(set(manifest.clips) - {'video_002/clip_001'}))
materialize_dataset(source, new_dataset, clip_ids=selected)
media_hashes = {}
for key in selected:
    clip = ClipManifest.load(source / manifest.clips[key].path)
    media_hashes[key] = {cam: sha256(clip.media_path(cam)) for cam in clip.camera_ids}
feature_sha = sha256(Path('/home/kamimura/projects/tennis-lab/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'))
detector_sha = sha256(root / 'ckpt/dino/checkpoint0029_4scale_swin.pth')
records = []
# Explicitly rejected by checkpoint-SHA audit; these are recomputed in v7.
rejected_features = {
    'video_001/clip_007': 'afc808ab2c7609e825883ec69125c7af08cd842bda5c132018896e563d4677ef',
    'video_002/clip_014': '70ebeabfb40b372489a6580ef85c8c15a1bb710b0a60140d35edea6ee9d6307d',
    'video_002/clip_016': 'afc808ab2c7609e825883ec69125c7af08cd842bda5c132018896e563d4677ef',
}
rejected_records = []

def copy_checked(src: Path, dst: Path, *, hardlink: bool = False) -> None:
    before = sha256(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256(dst) != before:
            raise ValueError(f'Existing destination differs: {dst}')
    elif hardlink:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    if before != sha256(src) or before != sha256(dst):
        raise ValueError(f'Cache changed while copying: {src}')
    records.append({'source': str(src.resolve()), 'destination': str(dst.resolve()),
                    'sha256': before, 'method': 'hardlink' if hardlink else 'copy'})

for key in selected:
    old_clip = ClipManifest.load(old_dataset / manifest.clips[key].path)
    new_clip = ClipManifest.load(new_dataset / manifest.clips[key].path)
    spec = load_dino_spec(old_clip.clip_dir)
    identity = {'script': 'src/tennis_scene/scripts/build_slcs_dataset.py',
                'checkpoint_sha256': feature_sha, 'video_sha256': media_hashes[key]}
    if key in rejected_features:
        marker = dino_dir(old_clip.clip_dir) / 'annotation.json'
        saved_hash = json.loads(marker.read_text())['generator']['checkpoint_sha256']
        if saved_hash != rejected_features[key]:
            raise ValueError(f'Rejected feature receipt changed: {marker}')
        if dino_dir(new_clip.clip_dir).exists():
            raise ValueError(f'Rejected RGB features must not exist in v7: {key}')
        rejected_records.append({'clip_id': key, 'receipt': str(marker.resolve()),
                                 'receipt_sha256': sha256(marker), 'saved_checkpoint_sha256': saved_hash,
                                 'expected_checkpoint_sha256': feature_sha,
                                 'action': 'not migrated; regenerate through normal build'})
        print(f'Explicitly rejected old RGB features: {key}', flush=True)
    else:
        if not validated_feature_cache(old_clip, spec, identity):
            raise ValueError(f'Missing expected RGB feature cache: {key}')
        feature_source = dino_dir(old_clip.clip_dir)
        feature_destination = dino_dir(new_clip.clip_dir)
        for entry in sorted(feature_source.glob('*.npz')):
            copy_checked(entry, feature_destination / entry.name, hardlink=True)
        copy_checked(feature_source / 'annotation.json', feature_destination / 'annotation.json')
        if not validated_feature_cache(new_clip, spec, identity):
            raise ValueError(f'Incomplete migrated RGB feature cache: {key}')
    old = old_observations / key
    new = new_observations / key
    if (old / 'court.npz').is_file() and (old / 'court.json').is_file():
        with np.load(old / 'court.npz', allow_pickle=False) as arrays:
            if not all(np.isfinite(arrays[name]).all() for name in ('keypoints', 'homographies')):
                raise ValueError(f'Invalid court arrays: {old}')
        copy_checked(old / 'court.npz', new / 'court.npz')
        copy_checked(old / 'court.json', new / 'court.json')
    for cam in old_clip.camera_ids:
        cache = old / f'{cam}_detections.npz'
        receipt = cache.with_suffix('.metadata.json')
        if not (cache.is_file() and receipt.is_file()):
            continue  # Pending old-run caches are explicitly not migrated.
        expected = {'schema_version': 1, 'video_sha256': media_hashes[key][cam],
                    'checkpoint_sha256': detector_sha, 'confidence': 0.3,
                    'short_side': 800, 'max_long_side': 1333, 'stride': 4,
                    'total_frames': old_clip.num_frames}
        if json.loads(receipt.read_text()) != expected:
            raise ValueError(f'Stale raw detections: {cache}')
        with np.load(cache, allow_pickle=False) as arrays:
            indices, offsets, boxes, scores = (arrays[n] for n in ('frame_indices','offsets','boxes','scores'))
            if (offsets.shape != (len(indices)+1,) or offsets[0] != 0
                or offsets[-1] != len(boxes) or scores.shape != (len(boxes),)
                or boxes.shape != (len(boxes),4) or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or np.any(np.diff(offsets) < 0)):
                raise ValueError(f'Invalid raw detection arrays: {cache}')
        copy_checked(cache, new / cache.name)
        copy_checked(receipt, new / receipt.name)
    print(f'Validated and seeded {key}', flush=True)

out = root / 'outputs/tennis_scene/analyze/meiji_cache_reuse/s42-001/receipt.json'
save_json_atomic({'purpose': 'reuse immutable raw DINO/court/RGB caches; regenerate all person observations and 3D teachers',
                  'source_dataset': str(old_dataset.resolve()), 'destination_dataset': str(new_dataset.resolve()),
                  'source_observations': str(old_observations.resolve()), 'destination_observations': str(new_observations.resolve()),
                  'script_sha256': sha256(Path(__file__)), 'clip_count': len(selected),
                  'records': records, 'rejected_features': rejected_records}, out)
print(json.dumps({'receipt': str(out), 'files': len(records), 'raw_detection_caches': sum(r['source'].endswith('_detections.npz') for r in records)}))
