"""Wiki pages that project the public person-model comparison sources."""

from __future__ import annotations

from collections.abc import Callable

from ..chart_spec import Chart
from ..sources import Sources, require
from ..table_spec import Table
from . import body, detection, methods, overview, pose

Asset = Table | Chart
PAGES: dict[str, Callable[[Sources], tuple[Asset, ...]]] = {
    "person-model-comparison-overview": overview.build,
    "person-model-comparison-methods": methods.build,
    "person-detector-and-ecpose-comparison": detection.build,
    "top-down-pose-model-comparison": pose.build,
    "body-recovery-fasthmr-vs-gvhmr": body.build,
}


def project(page: str, sources: Sources) -> tuple[Asset, ...]:
    assets = PAGES[page](sources)
    require(
        len({asset_data(asset).slug for asset in assets}) == len(assets),
        f"{page}: duplicate asset slug",
    )
    return assets


def asset_data(asset: Asset) -> Table:
    return asset if isinstance(asset, Table) else asset.data
