"""Source identity shared by the standard component and publication recipes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def source_commit() -> str:
    """Complete revision of the checkout that supplies the pipeline code."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def source_digests(source: Path) -> dict[str, str]:
    return {str(path.relative_to(source)): dual_sha256(path) for path in sorted(source.rglob("*.py"))}


def compatible_source_digests(
    sources: dict[str, str], *, ball_court_side: bool = False, tensorrt: bool = False,
    standard_recipe: bool = False, standard_people: bool = False, alternative_people: bool = False,
) -> dict[str, str]:
    """Keep the baseline identity for exact, compatible source changes."""
    if not alternative_people:
        sources = _compatible_patch(sources, "people_models_cache_compatibility.json")
    sources = _compatible_patch(sources, "multiview_scene_cache_compatibility.json")
    sources = _compatible_patch(sources, "event_detection_cache_compatibility.json")
    sources = _compatible_patch(sources, "pose_comparison_cache_compatibility.json")
    sources = _compatible_patch(sources, "multiview_tracking_cache_compatibility.json")
    sources = _compatible_patch(sources, "multiview_pose_cache_compatibility.json")
    if standard_recipe or standard_people:
        sources = _compatible_patch(sources, "ecpose_recipe_cache_compatibility.json")
    if standard_recipe:
        sources = _compatible_patch(sources, "standard_recipe_cache_compatibility.json")
    if not tensorrt:
        sources = _compatible_patch(sources, "batched_tensorrt_fp32_cache_compatibility.json")
        sources = _compatible_patch(sources, "tensorrt_fp32_cache_compatibility.json")
    if ball_court_side:
        sources = _compatible_patch(sources, "ball_court_side_cache_compatibility.json")
    return _compatible_patch(sources, "fp32_cache_compatibility.json")


def _compatible_patch(sources: dict[str, str], filename: str) -> dict[str, str]:
    manifest = Path(__file__).with_name(filename)
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    if any(sources.get(path) != versions["after"] for path, versions in files.items()):
        return sources

    # The files form one reviewed change. Keep all other source digests in
    # the identity so changes outside the patch still invalidate old caches.
    compatible = dict(sources)
    for path, versions in files.items():
        if versions["before"] is None:
            del compatible[path]
        else:
            compatible[path] = versions["before"]
    return compatible
