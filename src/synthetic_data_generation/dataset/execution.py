"""Execute immutable dataset plans across project and NHT Python runtimes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.dataset.pipeline import (
    DatasetPipelinePlan,
    PipelineCommand,
)
from src.synthetic_data_generation.rendering.nht.process import (
    NhtProcessBackend,
    ProcessResult,
)
from src.utils.io import save_json_atomic

PIPELINE_EXECUTION_SCHEMA = "tennis_synthetic_dataset_pipeline_execution_v1"


@dataclass(frozen=True)
class StageExecution:
    """One completed pipeline stage and its captured log files."""

    stage: str
    runtime: str
    returncode: int
    stdout_file: str
    stderr_file: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible execution record."""
        return {
            "stage": self.stage,
            "runtime": self.runtime,
            "returncode": self.returncode,
            "stdout_file": self.stdout_file,
            "stderr_file": self.stderr_file,
        }


class DatasetPipelineExecutor:
    """Run a plan sequentially and stop at the first failed stage."""

    def __init__(
        self,
        *,
        project_root: Path,
        nht_backend: NhtProcessBackend,
    ) -> None:
        self._project_root = project_root.resolve()
        self._nht_backend = nht_backend

    def _run_project(self, command: PipelineCommand) -> ProcessResult:
        if command.runtime != "project":
            raise ValueError("Project runner received a non-project stage.")
        argv = (sys.executable, "-m", command.module, *command.arguments)
        environment = os.environ.copy()
        previous_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(self._project_root)
            if not previous_pythonpath
            else f"{self._project_root}{os.pathsep}{previous_pythonpath}"
        )
        completed = subprocess.run(
            argv,
            cwd=self._project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        return ProcessResult(
            command=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def execute(
        self,
        plan: DatasetPipelinePlan,
        *,
        output_dir: Path,
    ) -> Path:
        """Execute all stages and atomically publish logs plus a manifest."""
        output_dir = output_dir.resolve()
        if output_dir.exists():
            raise FileExistsError(
                f"Pipeline execution refuses existing output: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                dir=output_dir.parent,
            )
        )
        records: list[StageExecution] = []
        try:
            plan.write(temporary / "plan.json")
            for index, command in enumerate(plan.commands):
                result = (
                    self._nht_backend.run(command)
                    if command.runtime == "nht"
                    else self._run_project(command)
                )
                stem = f"{index:02d}-{command.stage}"
                stdout_path = temporary / f"{stem}.stdout.log"
                stderr_path = temporary / f"{stem}.stderr.log"
                stdout_path.write_text(result.stdout, encoding="utf-8")
                stderr_path.write_text(result.stderr, encoding="utf-8")
                records.append(
                    StageExecution(
                        stage=command.stage,
                        runtime=command.runtime,
                        returncode=result.returncode,
                        stdout_file=stdout_path.name,
                        stderr_file=stderr_path.name,
                    )
                )
                result.raise_for_status()
            save_json_atomic(
                {
                    "schema": PIPELINE_EXECUTION_SCHEMA,
                    "dataset": plan.dataset,
                    "plan_fingerprint": plan.plan_fingerprint,
                    "stages": [record.to_dict() for record in records],
                    "complete": True,
                },
                temporary / "execution.json",
            )
            temporary.rename(output_dir)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return output_dir
