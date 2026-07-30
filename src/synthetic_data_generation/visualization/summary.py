"""Create a generic HTML summary from path-configured pipeline artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is malformed JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return value


def write_pipeline_visualization(manifest: PathPipelineManifest) -> Path:
    """Write paths, metrics, jobs, and renders as a portable HTML document."""
    sections = {
        "Configured paths": manifest.to_dict()["paths"],
        "Alignment metrics": _load_json_object(
            manifest.alignment_metrics,
            description="Alignment metrics",
        ),
        "Dataset plan": _load_json_object(
            manifest.dataset_plan,
            description="Dataset plan",
        ),
        "Render manifest": _load_json_object(
            manifest.render_manifest,
            description="Render manifest",
        ),
        "Quality metrics": _load_json_object(
            manifest.quality_metrics,
            description="Quality metrics",
        ),
    }
    blocks: list[str] = []
    for title, value in sections.items():
        formatted = html.escape(json.dumps(value, indent=2, ensure_ascii=False))
        blocks.append(
            f"<section><h2>{html.escape(title)}</h2><pre>{formatted}</pre></section>"
        )
    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Synthetic-data pipeline summary</title>",
            "<style>body{font-family:sans-serif;max-width:1000px;margin:auto}"
            "pre{overflow:auto;background:#f5f5f5;padding:1rem}</style>",
            "</head>",
            "<body>",
            "<h1>Synthetic-data pipeline summary</h1>",
            *blocks,
            "</body>",
            "</html>",
            "",
        ]
    )
    manifest.visualization.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.visualization.with_suffix(
        manifest.visualization.suffix + ".tmp"
    )
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(manifest.visualization)
    return manifest.visualization
