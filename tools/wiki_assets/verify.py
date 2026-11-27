"""Verify a generated bundle against the pinned sources and publication policy."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from .chart_spec import Bars, Scatter, pareto_frontier
from .markdown import snippet, verify_markdown
from .pages import Asset, asset_data
from .publication import scan_public_text, verify_pair
from .sources import Sources, obj, require
from .table_spec import Table
from .typst_runner import VERSION

SCHEMA = 1


def manifest_path(output: Path, page: str) -> Path:
    return output / "manifests" / page / "manifest.json"


def asset_paths(page: str, assets: tuple[Asset, ...]) -> set[Path]:
    return {
        Path("assets") / page / f"{asset_data(asset).slug}{ext}"
        for asset in assets
        for ext in (".svg", ".png")
    }


def file_digests(output: Path, page: str, assets: tuple[Asset, ...]) -> dict[str, str]:
    paths = {Path(f"{page}.md"), *asset_paths(page, assets)}
    digests = {}
    for path in sorted(paths):
        target = output / path
        require(
            target.is_file()
            and not target.is_symlink()
            and target.resolve().is_relative_to(output.resolve()),
            f"Missing generated file: {path}",
        )
        digests[path.as_posix()] = sha256(target.read_bytes()).hexdigest()
    return digests


def describe(
    page: str, sources: Sources, assets: tuple[Asset, ...]
) -> dict[str, object]:
    records = {}
    for asset in assets:
        table = asset_data(asset)
        record: dict[str, object] = {
            "kind": "table"
            if isinstance(asset, Table)
            else "scatter"
            if isinstance(asset, Scatter)
            else "bars",
            "columns": [c.heading for c in table.columns],
            "row_count": len(table.rows),
            "exclusions": {
                r.key: r.exclusion for r in table.rows if r.exclusion is not None
            },
            "rows": [
                {
                    "key": row.key,
                    "group": row.group,
                    "values": [
                        None if cell is None else str(cell) for cell in row.cells
                    ],
                    "display": list(cells),
                    "bold_columns": [
                        column.heading
                        for column, highlighted in zip(table.columns, bold, strict=True)
                        if highlighted
                    ],
                }
                for row, cells, bold in table.display_rows()
            ],
        }
        if isinstance(asset, Scatter):
            record["pareto"] = [row.key for row in pareto_frontier(asset)]
            record["axes"] = [table.columns[i].heading for i in asset.metrics]
            if asset.zoom is not None:
                record["zoom"] = [list(axis) for axis in asset.zoom]
        if isinstance(asset, Bars):
            record["axes"] = [table.columns[i].heading for i in asset.metrics]
        records[table.slug] = record
    return {
        "schema_version": SCHEMA,
        "page": page,
        "source_revision": sources.revision,
        "inputs": dict(sorted(sources.digests.items())),
        "assets": records,
    }


def check_asset_directory(output: Path, page: str, expected: set[Path]) -> None:
    directory = output / "assets" / page
    names = (
        {path.relative_to(output) for path in directory.iterdir()}
        if directory.is_dir()
        else set()
    )
    require(names <= expected, f"{page}: unexpected files in the asset directory")


def verify_bundle(
    output: Path, page: str, sources: Sources, assets: tuple[Asset, ...]
) -> dict[str, object]:
    path = manifest_path(output, page)
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(output.resolve()),
        f"{page}: missing manifest.json",
    )
    raw = path.read_text(encoding="utf-8")
    scan_public_text(raw, f"{page}/manifest.json")
    manifest = obj(json.loads(raw), "manifest")
    expected = describe(page, sources, assets)
    require(
        set(manifest) == set(expected) | {"files", "tools"},
        f"{page}: invalid manifest fields",
    )
    for key in ("schema_version", "page", "source_revision", "inputs"):
        require(
            manifest[key] == expected[key],
            f"{page}: manifest {key} differs from the pinned sources",
        )
    require(
        manifest["assets"] == expected["assets"],
        f"{page}: manifest numbers, states or rankings differ from sources",
    )
    tools = obj(manifest["tools"], "manifest tools")
    require(
        tools.get("typst") == VERSION
        and all(
            isinstance(tools.get(k), str)
            and re.fullmatch(r"[0-9][0-9.a-z+.-]*", str(tools[k])) is not None
            for k in ("matplotlib", "pillow")
        ),
        f"{page}: missing or invalid tool versions",
    )
    expected_assets = asset_paths(page, assets)
    check_asset_directory(output, page, expected_assets)
    md_path = output / f"{page}.md"
    svg_paths = verify_markdown(md_path)
    require(
        md_path.read_text(encoding="utf-8") == snippet(page, sources, assets),
        f"{page}: Markdown numbers, states or source references differ",
    )
    svg_expected = {
        output / target for target in expected_assets if target.suffix == ".svg"
    }
    require(
        svg_paths == svg_expected,
        f"{page}: expected SVG and PNG links for every figure",
    )
    for image in sorted(svg_paths):
        verify_pair(image)
    require(
        manifest["files"] == file_digests(output, page, assets),
        f"{page}: generated file digest differs",
    )
    return manifest
