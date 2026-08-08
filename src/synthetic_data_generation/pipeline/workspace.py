"""Fixed paths for one canonical mutable scene and its transactions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.pipeline.contracts import (
    StageDefinition,
    StageExecutionSummary,
)
from src.utils.configuration import PathResolver, PathRole


@dataclass(frozen=True, slots=True)
class SceneWorkspace:
    """All paths for one scene, resolved through the #688 data-root contract."""

    scene_id: str
    root: Path

    @classmethod
    def resolve(cls, resolver: PathResolver, scene_id: str) -> SceneWorkspace:
        """Resolve the sole workspace path for ``scene_id`` without experiment IDs."""
        root = resolver.resolve(
            PathRole.DATA,
            "synthetic_data_generation",
            "scenes",
            scene_id,
        )
        return cls(scene_id=scene_id, root=root)

    @property
    def run_manifest_path(self) -> Path:
        """Return the one mutable scene run manifest."""
        return self.root / "run.json"

    @property
    def resolved_config_path(self) -> Path:
        """Return the fixed resolved configuration path."""
        return self.root / "resolved-config.yaml"

    @property
    def transaction_root(self) -> Path:
        """Return the fixed same-filesystem root for all stage transactions."""
        return self.root / ".transactions"

    @property
    def exchange_probe_path(self) -> Path:
        """Return the fixed capability-probe location, never a run history path."""
        return self.transaction_root / ".exchange-probe"

    def owner_path(
        self,
        definition: StageDefinition[StageExecutionSummary],
    ) -> Path:
        """Resolve a stage's fixed owner directory beneath this scene."""
        candidate = (self.root / definition.owner_relative_path).resolve(strict=False)
        root = self.root.resolve(strict=False)
        if not candidate.is_relative_to(root) or candidate == root:
            raise ValueError(f"Stage owner escapes scene workspace: {candidate}")
        if candidate.is_relative_to(self.transaction_root.resolve(strict=False)):
            raise ValueError("Stage owner must not overlap the transaction authority.")
        return candidate

    def stage_transaction_path(
        self,
        definition: StageDefinition[StageExecutionSummary],
    ) -> Path:
        """Return the one fixed transaction directory for a stage."""
        candidate = self.transaction_root / definition.name.value
        if candidate.parent != self.transaction_root:
            raise ValueError("Stage transaction path escaped its fixed root.")
        return candidate

    def staging_path(
        self,
        definition: StageDefinition[StageExecutionSummary],
    ) -> Path:
        """Return the complete replacement snapshot outside the owner."""
        return self.stage_transaction_path(definition) / "snapshot"

    def validate_required_outputs(
        self,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Require every declared canonical output after semantic validation."""
        owner = self.owner_path(definition)
        missing = [
            str(path)
            for path in definition.required_outputs
            if not (owner / path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Stage {definition.name.value} is missing required outputs: {missing}"
            )

    def validate_stage_input(
        self,
        producer: StageDefinition[StageExecutionSummary],
        relative_path: Path,
    ) -> None:
        """Require one definition-authorized upstream artifact at its fixed owner."""
        path = self.owner_path(producer) / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Required {producer.name.value} input is missing: {relative_path}."
            )

    def invalidate_outputs(
        self,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Physically unpublish one complete fixed owner directory."""
        owner = self.owner_path(definition)
        if owner.is_symlink():
            owner.unlink()
        elif owner.is_dir():
            shutil.rmtree(owner)
        elif owner.exists():
            owner.unlink()


__all__ = ["SceneWorkspace"]
