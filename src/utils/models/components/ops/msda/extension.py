"""Load only the explicitly selected, preflighted DINO CUDA binary."""
from functools import lru_cache
from types import ModuleType
from typing import cast


@lru_cache(maxsize=1)
def require_extension() -> ModuleType:
    from src.submodules.models.dino.extension import (
        load_dino_extension,
        validate_dino_extension,
    )

    validate_dino_extension()
    return cast(ModuleType, load_dino_extension())
