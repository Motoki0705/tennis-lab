"""Opt-in CPU contract check against a real calibrated NHT workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


def test_real_subset_import_retains_full_geometry(tmp_path: Path) -> None:
    source_value = os.environ.get("NHT_SOURCE_WORKSPACE")
    if not source_value:
        pytest.skip("Set NHT_SOURCE_WORKSPACE to run the real-data CPU import test")
    assert source_value is not None
    source = Path(source_value)
    repository = Path(__file__).resolve().parents[3]
    nht = repository / "third_party/nht"
    executable = nht / ".venv/bin/nht-reconstruct"
    if not executable.is_file():
        pytest.fail("Install the NHT test environment before the real-data check")
    names = [f"frame_{round(i * 248 / 49):06d}.jpg" for i in range(50)]
    replacements = tmp_path / "replacements"
    replacements.mkdir()
    for name in names:
        shutil.copyfile(source / "frames/images" / name, replacements / name)
    config = yaml.safe_load((source / "resolved-config.yaml").read_text())
    trainer_python = config["nht_training"]["python"]
    config["nht_training"].update(
        image_names=names,
        max_steps=7000,
        trainer=str(nht / "gsplat/examples/simple_trainer_nht.py"),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    target = tmp_path / "imported"
    subprocess.run(
        [
            str(executable),
            "--scene-id",
            "appearance-contract-test",
            "--workspace",
            str(target),
            "--config",
            str(config_path),
            "--source-workspace",
            str(source),
            "--replacement-images",
            str(replacements),
            "--from-stage",
            "nht_training",
            "--prepare-only",
        ],
        check=True,
        env=dict(os.environ, PYTHONPATH=str(nht)),
        timeout=120,
    )
    assert sorted(path.name for path in (target / "frames/images").iterdir()) == names
    assert not (target / "3dgs").exists()
    metadata = json.loads((target / "import-provenance.json").read_text())
    assert metadata["full_image_count"] == 491
    assert len(metadata["validation_names"]) == 8
    assert not (target / "frames/raw").exists()
    # Native parser test: copied source RGB stands in only for this data-contract
    # check; these fixtures are never published or presented as generated images.
    dataset = tmp_path / "dataset"
    (dataset / "sparse").mkdir(parents=True)
    (dataset / "sparse/0").symlink_to(target / "sfm/model", target_is_directory=True)
    (dataset / "images").symlink_to(target / "frames/images", target_is_directory=True)
    (dataset / "images_2").symlink_to(
        target / "frames/training-images", target_is_directory=True
    )
    script = r"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
from datasets.colmap import Parser,Dataset
names=[f"frame_{round(i*248/49):06d}.jpg" for i in range(50)]
parser=Parser(sys.argv[2],factor=2,normalize=True,native_images_factor=True,image_names=names)
source=json.loads(Path(sys.argv[3]).read_text())
np.testing.assert_array_equal(parser.source_indices,[round(i*248/49) for i in range(50)])
np.testing.assert_allclose(parser.transform,source['sfm_to_scene'],rtol=0,atol=1e-12)
np.testing.assert_allclose(parser.scene_scale,source['scene_scale'],rtol=0,atol=1e-12)
train,val=Dataset(parser,split='train'),Dataset(parser,split='val')
assert len(train)==42 and len(val)==8
assert parser.full_image_count==491 and len(parser.image_names)==50
assert train[0]['image'].shape==val[0]['image'].shape
print(json.dumps({'selected':50,'train':len(train),'validation':len(val),'full':parser.full_image_count,'scene_scale':float(parser.scene_scale),'normalization_matches':True}))
"""
    completed = subprocess.run(
        [
            trainer_python,
            "-c",
            script,
            str(nht / "gsplat/examples"),
            str(dataset),
            str(source / "3dgs/scene-metadata.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["normalization_matches"] is True
    print(completed.stdout)
