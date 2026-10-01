"""Blind labeling retains reviewed switches, ignores synthetic frames and fails closed."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize('incomplete', [False, True])
def test_blind_materialization_uses_all_raw_observations_and_reviewed_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incomplete: bool,
) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    module = importlib.import_module('person_unseen_labels')
    clip = 'video_000/clip_002'
    report, reviews = tmp_path/'report', tmp_path/'reviews'
    (report/clip).mkdir(parents=True)
    (reviews/clip).mkdir(parents=True)
    (report/'plan.json').write_text(json.dumps({'records': [
        {'clip': clip, 'source': {'videos': [{'num_frames': 4}]}}]}))
    (report/clip/'person-execute.json').write_text('{}')
    (report/clip/'prediction.json').write_text('FORBIDDEN ASSOCIATION OUTPUT')
    raw = SimpleNamespace(track_ids=np.array([7]), observed=np.array([[True, False, True, True]]),
                          boxes_xyxy=np.tile([1, 2, 11, 22], (1, 4, 1)))
    monkeypatch.setattr(module, 'load_raw', lambda *_: {'cam0': raw})
    review: dict[str, Any] = {'clips': {clip: {'people': {
        'A': {'role': 'player', 'description': 'First person'},
        'B': {'role': 'non_player', 'description': 'Different person'}},
        'tracks': {'cam0': {} if incomplete else {7: [[0, 2, 'A'], [2, 4, 'B']]}}}}}
    (reviews/clip/'review.yaml').write_text(yaml.safe_dump(review))
    read_text = Path.read_text

    def forbid_prediction(path: Path, *args: Any, **kwargs: Any) -> str:
        assert path.name != 'prediction.json', 'Labeling must stay blind'
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', forbid_prediction)
    if incomplete:
        with pytest.raises(ValueError, match='unreviewed tracks'):
            module.labels(report, reviews)
        assert not (reviews/clip/'labels.json').exists()
    else:
        module.labels(report, reviews)
        result = ClipLabels.load(reviews/clip/'labels.json')
        assert result.cameras['cam0'].frames.tolist() == [0, 2, 3]
        assert result.cameras['cam0'].person_index.tolist() == [0, 1, 1]
        assert result.provenance['blind_to_association'] is True
        with pytest.raises(FileExistsError):
            module.labels(report, reviews)
