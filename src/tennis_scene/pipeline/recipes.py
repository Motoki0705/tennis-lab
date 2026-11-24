"""Names and storage paths for explicitly selected pipeline recipes."""

from pathlib import Path
from typing import Final, Literal

STANDARD_RECIPE: Final = "standard"
FASTHMR_RECIPE: Final = "fasthmr"
RecipeName = Literal["standard", "fasthmr"]


def parse_recipe(value: object) -> RecipeName:
    if value not in (STANDARD_RECIPE, FASTHMR_RECIPE):
        raise ValueError(f"Unknown pipeline recipe: {value!r}")
    return value


def recipe_directory(root: Path, recipe: RecipeName) -> Path:
    return root if recipe == STANDARD_RECIPE else root / "recipes" / recipe
