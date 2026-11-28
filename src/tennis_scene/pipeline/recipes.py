"""Names and storage paths for explicitly selected pipeline recipes."""

from pathlib import Path
from typing import Final, Literal

from src.tennis_scene.pipeline.model_assets import (
    LEGACY_PEOPLE_MODELS,
    DetectorName,
    PoseName,
)

STANDARD_RECIPE: Final = "standard"
FASTHMR_RECIPE: Final = "fasthmr"
ECPOSE_RECIPE: Final = "ecpose"
RecipeName = Literal["standard", "fasthmr", "ecpose"]


def parse_recipe(value: object) -> RecipeName:
    if value not in (STANDARD_RECIPE, FASTHMR_RECIPE, ECPOSE_RECIPE):
        raise ValueError(f"Unknown pipeline recipe: {value!r}")
    return value


def recipe_directory(root: Path, recipe: RecipeName, *, people_models: tuple[DetectorName, PoseName] | None = None) -> Path:
    directory = root if recipe == STANDARD_RECIPE else root / "recipes" / recipe
    if people_models is None or people_models == LEGACY_PEOPLE_MODELS:
        return directory
    return directory / "people-models" / "+".join(people_models)
