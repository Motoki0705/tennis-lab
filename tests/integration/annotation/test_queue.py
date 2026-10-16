"""Exercise the annotation adapter against the shipped queue registration protocol."""

from pathlib import Path

from src.tasks.base.annotation.artifacts import write_asset
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.execution.queue import enqueue, queue_state
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import file_digest
from src.utils.paths import PROJECT_ROOT


def test_shipped_queue_registers_annotation_stage_idempotently(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    script = tmp_path / "queue.sh"
    script.write_bytes(
        (
            PROJECT_ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"
        ).read_bytes()
    )
    config = QueueConfiguration(
        script=str(script),
        script_digest=file_digest(script),
        directory=str(layout.repository / ".training_queue"),
        session="synthetic-annotation",
        issue=1088,
    )
    # Registration never starts the queued command or reserves GPU capacity.
    launch = write_asset(layout, layout.path("campaigns", "launch.json"), b"fixture")
    first = enqueue(layout, config, launch, PROJECT_ROOT)
    assert enqueue(layout, config, launch, PROJECT_ROOT) == first
    queue = Path(config.directory)
    assert [path.name for path in (queue / "jobs").glob("*.job")] == [first]
    assert len(list((queue / "control/registrations").glob("*.receipt"))) == 1
    assert queue_state(config, launch)["state"] == "queued"
    (queue / "jobs" / first).rename(queue / "done" / first)
    assert enqueue(layout, config, launch, PROJECT_ROOT) == first
    assert not list((queue / "jobs").glob("*.job"))
    assert queue_state(config, launch)["state"] == "failed"
