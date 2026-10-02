import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from src.utils.paths import PROJECT_ROOT


def benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location('reserve', PROJECT_ROOT / 'tests/benchmarks/player_association_reserve.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reservation_chooses_longest_unused_per_video_and_records_exclusions() -> None:
    candidates = [{'clip_id': 'video_000/clip_000', 'num_frames': 1900},
                  {'clip_id': 'video_000/clip_002', 'num_frames': 1355},
                  {'clip_id': 'video_000/clip_009', 'num_frames': 744},
                  {'clip_id': 'video_001/clip_003', 'num_frames': 1607},
                  {'clip_id': 'video_001/clip_008', 'num_frames': 187}]
    selected, audit = benchmark().select_clips(candidates, {'video_000/clip_000'})
    assert selected == ['video_000/clip_002', 'video_001/clip_003']
    assert {row['reason'] for row in audit} == {'prior_person_design_or_calibration', 'below_600_frames',
        'longest_eligible_in_this_video', 'shorter_than_selected_clip_from_same_video'}
    with pytest.raises(ValueError, match='No >=600-frame unseen'):
        benchmark().select_clips(candidates, {'video_000/clip_000', 'video_001/clip_003'})


def test_evaluated_reservation_cannot_be_changed(tmp_path: Path) -> None:
    path = tmp_path / 'data/tennis_multivew/processed/meiji_3cam/dataset/annotations/player_association/unseen_protocol.json'
    path.parent.mkdir(parents=True)
    content = json.dumps({'evaluation_attempts': 1, 'tuning_complete': False, 'status': 'reserved_not_labelled'})
    path.write_text(content)
    with pytest.raises(ValueError, match='zero evaluations'):
        benchmark().reserve(tmp_path, tmp_path / 'report')
    assert path.read_text() == content and not (tmp_path / 'report').exists()
