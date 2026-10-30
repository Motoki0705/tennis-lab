"""Check the published data version and receipts of the three-update GPU smoke."""

import json
from pathlib import Path

from src.synthetic_data_generation.dataset.court.scale.publication import (
    load_dataset_version,
)
from src.utils.data.image_record_store import file_sha256

task_root = Path.cwd()
smoke = task_root / "tmp/court-scale-manifest-smoke"
outputs = smoke / "outputs/court_detection"
training = outputs / "train/dataset-version-smoke/s42"
evaluation_path = outputs / "evaluate/dataset-version-smoke-val/s42/evaluation.json"
evaluation = json.loads(evaluation_path.read_text())
condition = json.loads((training / "data-condition.json").read_text())
manifest = (
    task_root.parent
    / "court-scale-trajectory-strategies/tmp/court-scale-review/run-04/1bef5ceda3d453b269cbb99e6b0d422e1e8466ad4e89d527cfa12b67d361dcda.dataset.json"
)
version = load_dataset_version(manifest, verify_images=True)
counts = {
    scene["scene_id"]: scene["counts"]["qa_accepted"] for scene in version["scenes"]
}
assert counts == {"B00": 71, "B01": 80, "B02": 71, "B03": 76}
assert condition == evaluation["data_condition"]
assert condition["accepted_images"] == sum(counts.values()) == 298
assert condition["scene_images"] == counts
assert condition["data_version"] == version["dataset_version"]
assert condition["generation_seed"] == version["generation_seed"] == 42
assert condition["dataset_version_manifest"]["sha256"] == file_sha256(manifest)
assert evaluation["global_step"] == 3
assert evaluation["checkpoint_sha256"] == file_sha256(Path(evaluation["checkpoint"]))
assert list(evaluation["splits"]) == ["tennis_court_detector-val"]
val = evaluation["splits"]["tennis_court_detector-val"]
assert val["count"] == 2 and not val["pose_supervised"]
receipt = {
    "accepted_images": 298,
    "scene_images": counts,
    "generation_seed": 42,
    "training_seed": condition["training_seed"],
    "optimizer_updates": 3,
    "batch_size": 1,
    "compile": False,
    "input_long_side": 512,
    "source": evaluation["config"]["data"]["source"],
    "dataset_version_manifest_sha256": file_sha256(manifest),
    "scene_counts": {s["scene_id"]: s["counts"] for s in version["scenes"]},
    "train_batch_counts": evaluation["config"]["mixed"]["train_batch_counts"],
    "evaluation_split": "val",
    "evaluation_images": val["count"],
    "evaluation_pose_supervised": val["pose_supervised"],
    "evaluation_sha256": file_sha256(evaluation_path),
    "checkpoint_sha256": evaluation["checkpoint_sha256"],
    "limitation": "Three updates and two real validation images verify integration only; production batch 8 and full real evaluation are not measured.",
    "preflight_attempt_1": "Rejected absolute child paths before GPU startup; retry uses DATA/ARTIFACT-relative input paths.",
}
(smoke / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps(
        {
            k: receipt[k]
            for k in (
                "accepted_images",
                "scene_images",
                "optimizer_updates",
                "evaluation_images",
            )
        },
        indent=2,
    )
)
