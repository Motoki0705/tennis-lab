"""Compare inference port with unmodified upstream KPR modules on real dev crops.

No Torchreid installation. Minimal package namespaces load only inference files
from an explicit upstream checkout; dataset/training imports are never executed.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal.windows import gaussian

from src.tasks.person_tracking.archive import load_features
from src.tasks.player_association.appearance.kpr import (
    KprInference,
    load_checkpoint,
    prompt_heatmaps,
)
from src.tasks.player_association.appearance.sampling import crop
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return types.SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


def reference(upstream: Path, config: dict[str, Any]) -> Any:
    for name in ('torchreid', 'torchreid.models', 'torchreid.utils', 'torchreid.models.solider', 'torchreid.models.solider.backbones'):
        package = types.ModuleType(name)
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = package
    sys.modules['torchreid'].models = sys.modules['torchreid.models']  # type: ignore[attr-defined]

    def load(name: str, relative: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, upstream / relative)
        if spec is None or spec.loader is None:
            raise ValueError('Upstream inference module missing')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load('torchreid.utils.constants', 'torchreid/utils/constants.py')
    kpr = load('torchreid.models.kpr', 'torchreid/models/kpr.py')
    load('torchreid.models.promptable_transformer_backbone', 'torchreid/models/promptable_transformer_backbone.py')
    load('torchreid.models.solider.backbones.swin_transformer', 'torchreid/models/solider/backbones/swin_transformer.py')
    solider = load('torchreid.models.promptable_solider', 'torchreid/models/promptable_solider.py')

    def build_model(name: str, num_classes: int, *, config: Any, loss: str, pretrained: bool, **kwargs: Any) -> Any:
        if name != 'solider_swin_base_patch4_window7_224' or pretrained:
            raise ValueError('Parity only permits the fixed pretrained=False SOLIDER recipe')
        return solider.solider_swin(config=config, pretrained=False, **kwargs)

    sys.modules['torchreid.models'].build_model = build_model  # type: ignore[attr-defined]
    return kpr.KPR(num_classes=751, pretrained=False, loss='part_based', config=namespace(config)).eval()


def reference_prompts(upstream: Path, prompts: np.ndarray, negative: np.ndarray | None = None) -> np.ndarray:
    """Execute original Gaussian/COCO grouping/background transform methods."""
    scope: dict[str, Any] = {'np': np, 'torch': torch, 'cv2': cv2, 'gaussian': gaussian,
                             'OrderedDict': __import__('collections').OrderedDict}

    def include(path: str, names: set[str]) -> None:
        tree = ast.parse((upstream / path).read_text())
        nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)

    include('torchreid/utils/imagetools.py', {'gkern'})
    include('torchreid/data/datasets/keypoints_to_masks.py', {'KeypointsToMasks', 'rescale_keypoints'})
    class TransformBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    scope['MaskTransform'] = TransformBase
    include('torchreid/data/masks_transforms/mask_transform.py', {'MaskGroupingTransform', 'AddBackgroundMask'})
    names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
             'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    scope['COCO_KEYPOINTS_MAP'] = {name: i for i, name in enumerate(names)}
    include('torchreid/data/masks_transforms/coco_keypoints_transforms.py', {'CocoToSixBodyMasks'})
    gaussian_masks = scope['KeypointsToMasks'](mode='keypoints_gaussian', vis_thresh=.3, vis_continous=False)
    group = scope['CocoToSixBodyMasks']()
    background = scope['AddBackgroundMask']('sum', -1, .2)
    output = []
    for index, pose in enumerate(prompts):
        pose = pose.astype(np.float64).copy()
        outside = ((pose[:, :2] < 0) | (pose[:, :2] >= 1)).any(1)
        pose[outside] = 0
        masks = gaussian_masks(pose, (1, 1), (128, 384))
        masks = group.apply_to_mask(torch.from_numpy(masks))
        negative_masks = []
        if negative is not None:
            for other in negative[index]:
                other = other.astype(np.float64).copy()
                other[((other[:, :2] < 0) | (other[:, :2] >= 1)).any(1)] = 0
                negative_masks.append(gaussian_masks(other, (1, 1), (128, 384)))
        negative_map = torch.from_numpy(np.concatenate(negative_masks)).max(dim=0, keepdim=True)[0] \
            if negative_masks else torch.zeros((1, 384, 128))
        masks = torch.cat((negative_map, masks), 0)
        output.append(background.apply_to_mask(masks).float().numpy())
    result: np.ndarray = np.stack(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    weights = args.repo / 'ckpt/player_association/kpr/kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar'
    state, config = load_checkpoint(weights)
    old = reference(args.upstream, config)
    new = KprInference().eval()
    if set(old.state_dict()) != set(new.state_dict()) or set(new.state_dict()) != set(state):
        raise ValueError(f'KPR key mismatch: reference-only={set(old.state_dict()) - set(new.state_dict())}; port-only={set(new.state_dict()) - set(old.state_dict())}')
    old.load_state_dict(state, strict=True)
    new.load_state_dict(state, strict=True)
    manifest = json.loads((args.features / 'features.json').read_text())
    record = manifest['records']['clipreid_vitb16_market1501']['video_000/clip_000/cam0']
    if dual_sha256(Path(record['path'])) != record['sha256']:
        raise ValueError('Feature content changed')
    frames, provenance = load_features(Path(record['path']))
    video = provenance['source']
    if dual_sha256(Path(video['path'])) != video['sha256']:
        raise ValueError('Source video changed')
    reader = cv2.VideoCapture(video['path'])
    ok, pixels = reader.read()
    reader.release()
    if not ok:
        raise ValueError('Real dev frame unavailable')
    frame = frames[0]
    boxes = frame.boxes[:2]
    prompts = frame.poses[:2].astype(np.float64).copy()
    clipped = np.rint(boxes).clip([0, 0, 0, 0], [1920, 1080, 1920, 1080])
    prompts[..., :2] = (prompts[..., :2] - clipped[:, None, :2]) / (clipped[:, None, 2:] - clipped[:, None, :2])
    crops = torch.from_numpy(np.stack([crop(pixels, b, (384, 128)) for b in boxes]))
    negative = np.stack([np.delete(frame.poses, i, axis=0) for i in range(2)]).astype(np.float64)
    negative[..., :2] = (negative[..., :2] - clipped[:, None, None, :2]) / (clipped[:, None, None, 2:] - clipped[:, None, None, :2])
    trials = {}
    for mode, other in (('positive_only', None), ('all_other_detection_poses_negative', negative)):
        prompt_old, prompt_new = reference_prompts(args.upstream, prompts, other), prompt_heatmaps(prompts, other)
        if not np.array_equal(prompt_old, prompt_new):
            raise ValueError(f'Prompt mismatch: {np.max(np.abs(prompt_old - prompt_new))}')
        with torch.inference_mode():
            ref = old((crops - .5) / .5, prompt_masks=torch.from_numpy(prompt_old))
            embedded = F.normalize(torch.cat((ref[0]['bn_foreg'].unsqueeze(1), ref[0]['parts']), 1), p=2, dim=-1)
            visible = torch.cat((ref[1]['foreg'].unsqueeze(1), ref[1]['parts']), 1)
            masks = torch.cat((ref[5]['foreg'].unsqueeze(1), ref[5]['parts']), 1)
            actual = new((crops - .5) / .5, torch.from_numpy(prompt_new))
        trials[mode] = {name: float((a.float() - b.float()).abs().max()) for name, a, b in zip(
            ('embeddings', 'visibility', 'masks'), (embedded, visible, masks), actual, strict=True)}
    differences = {name: max(t[name] for t in trials.values()) for name in ('embeddings', 'visibility', 'masks')}
    result = {'status': 'ok' if max(differences.values()) <= 1e-6 else 'failed',
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip(),
        'source': video, 'frame': 0, 'rows': frame.rows[:2].tolist(), 'boxes': boxes.tolist(),
        'feature': record, 'weights': {'path': str(weights), 'sha256': dual_sha256(weights)},
        'state_keys_equal': True, 'state_keys': len(state), 'output_keys': ['embeddings', 'visibility', 'masks'],
        'max_abs_diff': differences, 'prompts_bitwise_equal': True, 'input_sha256': __import__('hashlib').sha256(crops.numpy().tobytes()).hexdigest(),
        'device': 'cpu', 'torch': torch.__version__, 'input_size': [384, 128], 'trials': trials,
        'port_files': {str(p): dual_sha256(p) for p in (
            Path(__file__).resolve().parents[2] / 'src/tasks/player_association/appearance/kpr.py',
            Path(__file__).resolve().parents[2] / 'src/tasks/player_association/appearance/kpr_vendor/heads.py',
            Path(__file__).resolve().parents[2] / 'src/tasks/player_association/appearance/solider_vendor/swin_transformer.py')},
        'upstream_files': {str(p.relative_to(args.upstream)): dual_sha256(p) for p in args.upstream.rglob('*.py')
                           if p.name in ('kpr.py', 'promptable_solider.py', 'promptable_transformer_backbone.py', 'swin_transformer.py', 'keypoints_to_masks.py', 'mask_transform.py', 'coco_keypoints_transforms.py', 'imagetools.py')}}
    write_json_atomic(args.report, result)
    print(json.dumps(result, indent=2))
    if result['status'] != 'ok':
        raise ValueError('KPR parity failed; do not enqueue features')


if __name__ == '__main__':
    main()
