"""Project V3 KP geometry and attach the production alpha/depth visibility rule."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.dataset.court.components.labels import (
    attach_renderer_visibility,
    project_court_semantics_v3,
)
from src.synthetic_data_generation.reconstruction.scene_export import validate_standard_scene_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'schema': 'court_probe_visibility_v1', 'alpha_threshold': 0.01, 'sample_radius_px': 1, 'views': []}
    for render_path in sorted(args.renders.glob('repeat-*/*/render.json')):
        manifest = json.loads(render_path.read_text())
        scene_id = manifest['scene_id']
        workspace = args.scenes / scene_id
        export = validate_standard_scene_export(workspace / 'reconstruction/export/scene.json', without_source_images=True)
        alignment = load_alignment_result(workspace / 'alignment/alignment.json')
        for render in manifest['renders']:
            camera = export.camera(render['camera_id'])
            metric_camera = replace(camera, camera_to_scene=alignment.metric_adapter.metric_from_nht_camera(camera.camera_to_scene))
            geometry = project_court_semantics_v3(metric_camera, alignment.layout)
            visible = attach_renderer_visibility(
                geometry,
                alpha=np.load(render_path.parent / render['alpha']),
                depth=np.load(render_path.parent / render['depth']) / alignment.metric_adapter.nht_scene_units_per_metre,
            )
            result['views'].append({
                'scene_id': scene_id,
                'repeat': render_path.parent.parent.name,
                'camera_id': camera.camera_id,
                'points': [
                    {'court_id': court.court_instance_id, 'class_id': cls.class_id, 'physical_index': point.physical_index,
                     'uv': list(point.uv), 'depth_m': point.camera_depth_m, 'in_front': point.in_front,
                     'in_frame': point.in_frame, 'renderer_visible': point.renderer_visible}
                    for court in visible.courts for cls in court.classes for point in cls.points
                ],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'views': len(result['views']), 'points': sum(len(v['points']) for v in result['views']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
