from copy import deepcopy
from pathlib import Path
from runpy import run_path

import pytest
import yaml


@pytest.fixture
def helper():
    return run_path(
        "knowledge/runs/run-court-scene-diversity-colab-s42/colab_campaign.py"
    )


@pytest.mark.parametrize("condition", [1, 2, 3, 4])
def test_portable_config_preserves_frozen_training_semantics(helper, condition):
    original = yaml.safe_load((helper["PREP"] / f"scene-{condition}.yaml").read_text())
    before = deepcopy(original)
    result = helper["portable_config"](original, condition, Path("/content/tennis-lab"))
    assert original == before
    for key in ("model", "loss", "data", "mixed"):
        assert result[key] == original[key]
    training = deepcopy(result["training"])
    training["checkpoint"]["save_top_k"] = 0
    assert training == original["training"]
    assert result["run"]["seed"] == 42
    assert result["run"]["resume"] is result["run"]["init_weights"] is None
    assert result["training"]["step_schedule"]["max_steps"] == 33160
    assert result["training"]["warmup_steps"] == 4974
    assert result["training"]["checkpoint"]["save_top_k"] == -1
    assert result["paths"]["data_root"] == "/content/tennis-lab/data"
    assert result["run"]["artifact_store"]["remote_root"].endswith(
        f"/scene-{condition}-s42"
    )


@pytest.mark.parametrize("field", ["loss", "steps", "batch", "seed", "resume", "root"])
def test_portability_audit_rejects_unapproved_changes(helper, field):
    original = yaml.safe_load((helper["PREP"] / "scene-1.yaml").read_text())
    root = Path("/content/tennis-lab")
    result = helper["portable_config"](original, 1, root)
    if field == "loss":
        result["loss"]["pose"]["enabled"] = False
    elif field == "steps":
        result["training"]["step_schedule"]["max_steps"] = 100
    elif field == "batch":
        result["data"]["batch_size"] = 1
    elif field == "seed":
        result["run"]["seed"] = 43
    elif field == "resume":
        result["run"]["resume"] = "previous-condition.ckpt"
    else:
        result["paths"]["data_root"] = "/tmp/fake-data"
    with pytest.raises(ValueError):
        helper["validate_conversion"](original, result, 1, root)


def test_portable_plan_is_pinned_and_refuses_replacement(helper, tmp_path):
    helper["prepare"](tmp_path, Path("/content/tennis-lab"))
    helper["prepare"](tmp_path, Path("/content/tennis-lab"))
    p = tmp_path / "scene-2.yaml"
    config = yaml.safe_load(p.read_text())
    config["run"]["seed"] = 1
    p.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="overwrite"):
        helper["prepare"](tmp_path, Path("/content/tennis-lab"))
