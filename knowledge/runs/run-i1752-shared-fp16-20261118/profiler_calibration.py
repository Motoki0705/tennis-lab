"""CPU paired artifact verification, not an end-to-end GPU speedup estimate."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'knowledge/runs/run-i1751-runtime-gpu-a5'))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU calibration requires CUDA_VISIBLE_DEVICES empty')
    from benchmark import Observer
    from hydra import compose, initialize_config_dir
    from src.tennis_scene.configuration import parse_video_batch_config
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
    from src.tennis_scene.scripts.generate_dataset import _compose_pipeline_config

    spec = importlib.util.spec_from_file_location('qualification', Path(__file__).with_name('acceptance.py'))
    qualification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)
    root = REPO / 'outputs/tennis_scene/evaluate/video-batch-shared-fp16-20261118'
    with initialize_config_dir(config_dir=str(REPO / 'src/tennis_scene/configs'), version_base='1.3'):
        cfg = compose(config_name='run_video_batch', overrides=[a for a in qualification.arguments(root.resolve()) if not a.startswith('hydra.')])
        batch = parse_video_batch_config(cfg)
        _, runtime = _compose_pipeline_config(batch.generation, config_name=batch.pipeline_config)
    orchestrator = TennisSceneOrchestrator(runtime)
    court = orchestrator.prepare_video_court()
    side = orchestrator.prepare_video_court_side(court)
    report = {'scope': 'One verify of identical already-loaded artifacts, no inference, no GPU. AB/BA counterbalanced order, two observations each; not a whole-pipeline correction.',
              'court_digest': court.digest, 'side_digest': side.digest, 'measurements': []}
    with tempfile.TemporaryDirectory() as temp:
        for name, artifact in [('court', court), ('court_side', side)]:
            for index, profile in enumerate((False, True, True, False)):
                observer = Observer(Path(temp))
                try:
                    if profile:
                        sys.setprofile(observer.profile)
                    started = time.perf_counter()
                    artifact.verify()
                    seconds = time.perf_counter() - started
                finally:
                    sys.setprofile(None)
                report['measurements'].append({'artifact': name, 'global_python_profiler': profile, 'order': index, 'seconds': seconds})
    Path(__file__).with_name('profiler-calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
