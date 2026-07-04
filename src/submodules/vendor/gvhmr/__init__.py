"""Vendored GVHMR inference code.

See ``README.md`` in this package for provenance and modifications.
"""

from src.submodules.vendor.gvhmr.pipeline import (
    GvhmrDemoModel,
    Pipeline,
    build_gvhmr_demo_model,
)

__all__ = ["GvhmrDemoModel", "Pipeline", "build_gvhmr_demo_model"]
