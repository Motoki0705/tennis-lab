"""Fixed path resolution for one canonical mutable scene workspace."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.pipeline.contracts import StageSpec
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

    def owner_path(self, spec: StageSpec) -> Path:
        """Resolve a stage's fixed owner directory beneath this scene."""
        candidate = (self.root / spec.owner_relative_path).resolve(strict=False)
        root = self.root.resolve(strict=False)
        if not candidate.is_relative_to(root) or candidate == root:
            raise ValueError(f"Stage owner escapes scene workspace: {candidate}")
        return candidate

    def staging_path(self, spec: StageSpec) -> Path:
        """Return the stage-local mutable staging directory."""
        return self.owner_path(spec) / "staging"

    def validate_required_outputs(self, spec: StageSpec) -> None:
        """Require every declared canonical output after semantic validation."""
        owner = self.owner_path(spec)
        missing = [str(path) for path in spec.required_outputs if not (owner / path).exists()]
        if missing:
            raise FileNotFoundError(
                f"Stage {spec.name.value} is missing required outputs: {missing}"
            )

    def invalidate_outputs(self, spec: StageSpec) -> None:
        """Physically unpublish one stage's outputs and all attempt-local staging."""
        owner = self.owner_path(spec)
        for relative in (*spec.required_outputs, Path("staging")):
            target = owner / relative
            resolved = target.resolve(strict=False)
            if not resolved.is_relative_to(owner.resolve(strict=False)):
                raise ValueError(f"Refusing to invalidate escaped stage path: {resolved}")
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()


__all__ = ["SceneWorkspace"]
