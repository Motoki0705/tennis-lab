"""Overview:
Create a round-level Grounding DINO LoRA training dataset from one guardrail and one pseudo-label directory.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py dry_run=true output_dir=data/dino_workflow/training_sets/round_002 pseudo_dir=data/dino_workflow/pseudo/round_002

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/build_round_dataset.yaml`.
    - The output always exposes only `guardrail/` and `pseudo/` dataset directories for round training.
    - Court keypoints, roles, officials, ball boys, and ball labels stay mixed in one pseudo directory per round.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


IMAGE_KEYS = ("image_path", "image_file", "file_name", "image", "path", "absolute_image_path")


@dataclass(slots=True)
class DatasetSummary:
    """Summary of one source dataset directory."""

    kind: str
    dataset_dir: Path | None
    annotation_file: Path | None
    annotation_file_name: str
    row_count: int
    annotation_count: int
    image_count: int
    task_counts: Counter[str]
    label_counts: Counter[str]
    query_counts: Counter[str]
    missing_images: list[dict[str, Any]]
    exists: bool
    empty: bool

    def to_manifest(self, *, max_missing_preview: int) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir is not None else None,
            "annotation_file": str(self.annotation_file) if self.annotation_file is not None else None,
            "annotation_file_name": self.annotation_file_name,
            "exists": self.exists,
            "empty": self.empty,
            "row_count": self.row_count,
            "annotation_count": self.annotation_count,
            "image_count": self.image_count,
            "task_counts": dict(sorted(self.task_counts.items())),
            "label_counts": dict(sorted(self.label_counts.items())),
            "query_counts": dict(sorted(self.query_counts.items())),
            "missing_image_count": len(self.missing_images),
            "missing_images_preview": self.missing_images[:max_missing_preview],
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(to_absolute_path(text)).resolve()


def resolve_existing_dir(value: Any, *, field_name: str, required: bool) -> Path | None:
    path = optional_path(value)
    if path is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if not path.is_dir():
        if required:
            raise FileNotFoundError(f"{field_name} is not a directory: {path}")
        return None
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def annotation_candidates(dataset_dir: Path, preferred_name: str, *, dataset_kind: str) -> list[Path]:
    fallback_names = ["selected_annotations.jsonl", "annotations.jsonl"] if dataset_kind == "pseudo" else ["annotations.jsonl"]
    names = [preferred_name, *fallback_names]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        if name is None:
            continue
        path = (dataset_dir / str(name)).resolve()
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates


def find_annotation_file(dataset_dir: Path, preferred_name: str, *, dataset_kind: str) -> Path | None:
    for candidate in annotation_candidates(dataset_dir, preferred_name, dataset_kind=dataset_kind):
        if candidate.is_file():
            return candidate
    return None


def normalize_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = row.get("annotations")
    if isinstance(annotations, list):
        return [dict(item) for item in annotations if isinstance(item, dict)]
    if isinstance(annotations, dict):
        return [dict(annotations)]

    row_level_excludes = {
        *IMAGE_KEYS,
        "width",
        "height",
        "split",
        "label_source",
        "weight",
        "source",
        "guardrail",
        "source_annotation_file",
        "original_image_path",
        "pseudo_round",
    }
    annotation = {key: value for key, value in row.items() if key not in row_level_excludes}
    return [annotation] if annotation else []


def count_row_fields(
    row: dict[str, Any],
    *,
    task_counts: Counter[str],
    label_counts: Counter[str],
    query_counts: Counter[str],
) -> int:
    annotations = normalize_annotations(row)
    if not annotations:
        if row.get("task") is not None:
            task_counts[str(row["task"])] += 1
        if row.get("label") is not None:
            label_counts[str(row["label"])] += 1
        if row.get("query") is not None:
            query_counts[str(row["query"])] += 1
        return 0

    for annotation in annotations:
        task = annotation.get("task", row.get("task"))
        label = annotation.get("label", row.get("label"))
        query = annotation.get("query", row.get("query"))
        if task is not None:
            task_counts[str(task)] += 1
        if label is not None:
            label_counts[str(label)] += 1
        if query is not None:
            query_counts[str(query)] += 1
    return len(annotations)


def row_image_value(row: dict[str, Any]) -> str | None:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def resolve_image_path(image_value: str, *, dataset_dir: Path, annotation_file: Path) -> Path | None:
    raw = Path(image_value).expanduser()
    if raw.is_absolute() and raw.is_file():
        return raw.resolve()

    candidates = [
        (dataset_dir / raw).resolve(),
        (annotation_file.parent / raw).resolve(),
        (dataset_dir / "images" / raw).resolve(),
        (annotation_file.parent / "images" / raw).resolve(),
        (dataset_dir / "images" / raw.name).resolve(),
        (annotation_file.parent / "images" / raw.name).resolve(),
        Path(to_absolute_path(str(raw))).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def summarize_dataset(
    *,
    kind: str,
    dataset_dir: Path | None,
    preferred_annotation_file: str,
    validate_images: bool,
) -> DatasetSummary:
    if dataset_dir is None:
        return DatasetSummary(
            kind=kind,
            dataset_dir=None,
            annotation_file=None,
            annotation_file_name=preferred_annotation_file,
            row_count=0,
            annotation_count=0,
            image_count=0,
            task_counts=Counter(),
            label_counts=Counter(),
            query_counts=Counter(),
            missing_images=[],
            exists=False,
            empty=True,
        )

    annotation_file = find_annotation_file(
        dataset_dir,
        preferred_annotation_file,
        dataset_kind=kind,
    )
    if annotation_file is None:
        return DatasetSummary(
            kind=kind,
            dataset_dir=dataset_dir,
            annotation_file=None,
            annotation_file_name=preferred_annotation_file,
            row_count=0,
            annotation_count=0,
            image_count=0,
            task_counts=Counter(),
            label_counts=Counter(),
            query_counts=Counter(),
            missing_images=[],
            exists=True,
            empty=True,
        )

    rows = read_jsonl(annotation_file)
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    image_values: set[str] = set()
    missing_images: list[dict[str, Any]] = []
    annotation_count = 0

    for row_index, row in enumerate(rows, start=1):
        annotation_count += count_row_fields(
            row,
            task_counts=task_counts,
            label_counts=label_counts,
            query_counts=query_counts,
        )
        image_value = row_image_value(row)
        if image_value is None:
            missing_images.append(
                {
                    "row_index": row_index,
                    "image": None,
                    "reason": "missing image field",
                }
            )
            continue
        image_values.add(image_value)
        if validate_images and resolve_image_path(
            image_value,
            dataset_dir=dataset_dir,
            annotation_file=annotation_file,
        ) is None:
            missing_images.append(
                {
                    "row_index": row_index,
                    "image": image_value,
                    "reason": "image file not found",
                }
            )

    return DatasetSummary(
        kind=kind,
        dataset_dir=dataset_dir,
        annotation_file=annotation_file,
        annotation_file_name=annotation_file.name,
        row_count=len(rows),
        annotation_count=annotation_count,
        image_count=len(image_values),
        task_counts=task_counts,
        label_counts=label_counts,
        query_counts=query_counts,
        missing_images=missing_images,
        exists=True,
        empty=len(rows) == 0,
    )


def infer_pseudo_round(pseudo_dir: Path | None, output_dir: Path) -> dict[str, Any]:
    for path in (pseudo_dir, output_dir):
        if path is None:
            continue
        candidates = [path.name, path.as_posix()]
        for value in candidates:
            match = re.search(r"round[_-]?0*(\d+)", value, flags=re.IGNORECASE)
            if match:
                return {"name": f"round_{int(match.group(1)):03d}", "index": int(match.group(1))}
    return {"name": None, "index": None}


def validate_source_summaries(
    *,
    guardrail_summary: DatasetSummary,
    pseudo_summary: DatasetSummary,
    allow_empty_pseudo: bool,
    validate_images: bool,
    max_missing_preview: int,
) -> None:
    if guardrail_summary.annotation_file is None:
        raise FileNotFoundError(
            f"Guardrail annotations not found in {guardrail_summary.dataset_dir}; expected {guardrail_summary.annotation_file_name}"
        )
    if guardrail_summary.empty:
        raise ValueError(f"Guardrail annotations are empty: {guardrail_summary.annotation_file}")
    if validate_images and guardrail_summary.missing_images:
        preview = guardrail_summary.missing_images[:max_missing_preview]
        raise FileNotFoundError(f"Guardrail contains missing image references: {preview}")

    pseudo_missing_or_empty = pseudo_summary.annotation_file is None or pseudo_summary.empty
    if pseudo_missing_or_empty and not allow_empty_pseudo:
        raise FileNotFoundError(
            "Pseudo annotations are missing or empty. "
            f"pseudo_dir={pseudo_summary.dataset_dir}, expected={pseudo_summary.annotation_file_name}. "
            "Set allow_empty_pseudo=true to build a guardrail-only round dataset."
        )
    if validate_images and pseudo_summary.missing_images and not pseudo_missing_or_empty:
        preview = pseudo_summary.missing_images[:max_missing_preview]
        raise FileNotFoundError(f"Pseudo contains missing image references: {preview}")


def remove_existing_output(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"output_dir already exists and overwrite=false: {output_dir}")
    if output_dir.is_symlink() or output_dir.is_file():
        output_dir.unlink()
    else:
        shutil.rmtree(output_dir)


def materialize_dataset_dir(
    *,
    source_dir: Path,
    destination_dir: Path,
    link_mode: str,
    symlink_fallback_to_copy: bool,
) -> dict[str, Any]:
    if link_mode not in {"symlink", "copy", "hardlink"}:
        raise ValueError("link_mode must be one of: symlink, copy, hardlink")

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        try:
            os.symlink(source_dir, destination_dir, target_is_directory=True)
            return {
                "source_dir": str(source_dir),
                "destination_dir": str(destination_dir),
                "requested_mode": link_mode,
                "actual_mode": "symlink",
                "fallback_used": False,
            }
        except OSError as exc:
            if not symlink_fallback_to_copy:
                raise RuntimeError(f"Failed to symlink {source_dir} -> {destination_dir}") from exc
            shutil.copytree(source_dir, destination_dir, copy_function=shutil.copy2, symlinks=False)
            return {
                "source_dir": str(source_dir),
                "destination_dir": str(destination_dir),
                "requested_mode": link_mode,
                "actual_mode": "copy",
                "fallback_used": True,
                "fallback_reason": str(exc),
            }

    if link_mode == "copy":
        shutil.copytree(source_dir, destination_dir, copy_function=shutil.copy2, symlinks=False)
        return {
            "source_dir": str(source_dir),
            "destination_dir": str(destination_dir),
            "requested_mode": link_mode,
            "actual_mode": "copy",
            "fallback_used": False,
        }

    shutil.copytree(source_dir, destination_dir, copy_function=os.link, symlinks=False)
    return {
        "source_dir": str(source_dir),
        "destination_dir": str(destination_dir),
        "requested_mode": link_mode,
        "actual_mode": "hardlink",
        "fallback_used": False,
    }


def create_empty_pseudo_dir(destination_dir: Path, *, annotation_file_name: str) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    (destination_dir / "images").mkdir(parents=True, exist_ok=True)
    configured_annotation = destination_dir / annotation_file_name
    configured_annotation.touch()
    if annotation_file_name not in {"selected_annotations.jsonl", "annotations.jsonl"}:
        (destination_dir / "selected_annotations.jsonl").touch()
    write_json(
        destination_dir / "manifest.json",
        {
            "created_at": now_iso(),
            "empty_pseudo": True,
            "annotation_file": annotation_file_name,
        },
    )
    return {
        "source_dir": None,
        "destination_dir": str(destination_dir),
        "requested_mode": "empty",
        "actual_mode": "empty",
        "fallback_used": False,
    }


def combined_counter(*counters: Counter[str]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for counter in counters:
        total.update(counter)
    return dict(sorted(total.items()))


def build_manifest(
    *,
    cfg: DictConfig,
    output_dir: Path,
    guardrail_summary: DatasetSummary,
    pseudo_summary: DatasetSummary,
    materialization: dict[str, Any],
) -> dict[str, Any]:
    pseudo_round = infer_pseudo_round(pseudo_summary.dataset_dir, output_dir)
    max_missing_preview = int(cfg.max_missing_preview)
    return {
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "source_paths": {
            "guardrail_dir": str(guardrail_summary.dataset_dir) if guardrail_summary.dataset_dir else None,
            "pseudo_dir": str(pseudo_summary.dataset_dir) if pseudo_summary.dataset_dir else None,
        },
        "link_mode": str(cfg.link_mode),
        "materialization": materialization,
        "allow_empty_pseudo": bool(cfg.allow_empty_pseudo),
        "validate_images": bool(cfg.validate_images),
        "pseudo_round": pseudo_round,
        "counts": {
            "guardrail_rows": guardrail_summary.row_count,
            "pseudo_rows": pseudo_summary.row_count,
            "total_rows": guardrail_summary.row_count + pseudo_summary.row_count,
            "guardrail_annotations": guardrail_summary.annotation_count,
            "pseudo_annotations": pseudo_summary.annotation_count,
            "total_annotations": guardrail_summary.annotation_count + pseudo_summary.annotation_count,
            "guardrail_images": guardrail_summary.image_count,
            "pseudo_images": pseudo_summary.image_count,
            "total_image_refs": guardrail_summary.image_count + pseudo_summary.image_count,
        },
        "task_counts": combined_counter(guardrail_summary.task_counts, pseudo_summary.task_counts),
        "label_counts": combined_counter(guardrail_summary.label_counts, pseudo_summary.label_counts),
        "query_counts": combined_counter(guardrail_summary.query_counts, pseudo_summary.query_counts),
        "datasets": {
            "guardrail": guardrail_summary.to_manifest(max_missing_preview=max_missing_preview),
            "pseudo": pseudo_summary.to_manifest(max_missing_preview=max_missing_preview),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def write_round_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    pseudo_name = manifest["pseudo_round"]["name"] or output_dir.name
    text = f"""# Grounding DINO LoRA Round Dataset

This directory is a self-contained training input for `{pseudo_name}`.

Use it with:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py \\
  guardrail_dir={output_dir / "guardrail"} \\
  pseudo_dir={output_dir / "pseudo"}
```

Layout:

```text
{output_dir.name}/
  guardrail/
  pseudo/
  manifest.json
  README.md
```

`guardrail/` contains trusted supervised annotations. `pseudo/` contains the
reviewed pseudo labels for this round, or an empty annotation file when the
round was intentionally built with `allow_empty_pseudo=true`.
"""
    write_text(output_dir / "README.md", text)


def archive_pack_summary(cfg: DictConfig, *, input_root: Path, output_path: Path) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from archive_pack import archive_pack

    archive_cfg = OmegaConf.create(
        {
            "input_paths": [
                str(input_root / "guardrail"),
                str(input_root / "pseudo"),
                str(input_root / "manifest.json"),
                str(input_root / "README.md"),
            ],
            "output_archive": str(output_path),
            "base_dir": str(input_root),
            "include_manifest": True,
            "exclude_globs": ["__pycache__", "__pycache__/**", "*.pyc", ".DS_Store"],
            "overwrite": bool(cfg.archive.overwrite),
            "dry_run": False,
            "compression_level": int(cfg.archive.compression_level),
            "zstd_binary": str(cfg.archive.zstd_binary),
            "max_summary_files": int(cfg.archive.max_summary_files),
        }
    )
    return archive_pack(archive_cfg)


def stage_round_dataset_for_archive(output_dir: Path, staging_dir: Path) -> None:
    shutil.copytree(output_dir / "guardrail", staging_dir / "guardrail", copy_function=shutil.copy2, symlinks=False)
    shutil.copytree(output_dir / "pseudo", staging_dir / "pseudo", copy_function=shutil.copy2, symlinks=False)
    shutil.copy2(output_dir / "manifest.json", staging_dir / "manifest.json")
    shutil.copy2(output_dir / "README.md", staging_dir / "README.md")


def archive_round_dataset(
    cfg: DictConfig,
    output_dir: Path,
    *,
    materialization: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(cfg.archive.enabled):
        return None

    output_path = optional_path(cfg.archive.output_path)
    if output_path is None:
        output_path = output_dir.parent / f"{output_dir.name}.tar.zst"

    uses_symlink = any(
        value.get("actual_mode") == "symlink"
        for value in materialization.values()
        if isinstance(value, dict)
    )
    if not uses_symlink:
        summary = archive_pack_summary(cfg, input_root=output_dir, output_path=output_path)
        summary["staged_from_symlinks"] = False
        return summary

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}_archive_", dir=output_dir.parent) as temp_dir:
        staging_dir = Path(temp_dir)
        stage_round_dataset_for_archive(output_dir, staging_dir)
        summary = archive_pack_summary(cfg, input_root=staging_dir, output_path=output_path)
        summary["staged_from_symlinks"] = True
        return summary


def summary_from_manifest(
    manifest: dict[str, Any],
    *,
    dry_run: bool,
    archive_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = {
        "dry_run": dry_run,
        "output_dir": manifest["output_dir"],
        "source_paths": manifest["source_paths"],
        "link_mode": manifest["link_mode"],
        "materialization": manifest["materialization"],
        "pseudo_round": manifest["pseudo_round"],
        "counts": manifest["counts"],
        "task_counts": manifest["task_counts"],
        "label_counts": manifest["label_counts"],
        "missing_image_counts": {
            "guardrail": manifest["datasets"]["guardrail"]["missing_image_count"],
            "pseudo": manifest["datasets"]["pseudo"]["missing_image_count"],
        },
    }
    if archive_summary is not None:
        summary["archive"] = archive_summary
    return summary


def build_round_dataset(cfg: DictConfig) -> dict[str, Any]:
    guardrail_dir = resolve_existing_dir(cfg.guardrail_dir, field_name="guardrail_dir", required=True)
    pseudo_dir = resolve_existing_dir(cfg.pseudo_dir, field_name="pseudo_dir", required=False)
    assert guardrail_dir is not None
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()

    guardrail_summary = summarize_dataset(
        kind="guardrail",
        dataset_dir=guardrail_dir,
        preferred_annotation_file=str(cfg.guardrail_annotations_file),
        validate_images=bool(cfg.validate_images),
    )
    pseudo_summary = summarize_dataset(
        kind="pseudo",
        dataset_dir=pseudo_dir,
        preferred_annotation_file=str(cfg.pseudo_annotations_file),
        validate_images=bool(cfg.validate_images),
    )
    validate_source_summaries(
        guardrail_summary=guardrail_summary,
        pseudo_summary=pseudo_summary,
        allow_empty_pseudo=bool(cfg.allow_empty_pseudo),
        validate_images=bool(cfg.validate_images),
        max_missing_preview=int(cfg.max_missing_preview),
    )

    pseudo_is_empty = pseudo_summary.annotation_file is None or pseudo_summary.empty
    materialization: dict[str, Any] = {
        "guardrail": {
            "source_dir": str(guardrail_dir),
            "destination_dir": str(output_dir / "guardrail"),
            "requested_mode": str(cfg.link_mode),
            "actual_mode": "planned",
            "fallback_used": False,
        },
        "pseudo": {
            "source_dir": str(pseudo_dir) if pseudo_dir is not None else None,
            "destination_dir": str(output_dir / "pseudo"),
            "requested_mode": "empty" if pseudo_is_empty else str(cfg.link_mode),
            "actual_mode": "planned_empty" if pseudo_is_empty else "planned",
            "fallback_used": False,
        },
    }
    dry_manifest = build_manifest(
        cfg=cfg,
        output_dir=output_dir,
        guardrail_summary=guardrail_summary,
        pseudo_summary=pseudo_summary,
        materialization=materialization,
    )
    if bool(cfg.dry_run):
        return summary_from_manifest(dry_manifest, dry_run=True, archive_summary=None)

    remove_existing_output(output_dir, overwrite=bool(cfg.overwrite))
    output_dir.mkdir(parents=True, exist_ok=True)

    materialization["guardrail"] = materialize_dataset_dir(
        source_dir=guardrail_dir,
        destination_dir=output_dir / "guardrail",
        link_mode=str(cfg.link_mode),
        symlink_fallback_to_copy=bool(cfg.symlink_fallback_to_copy),
    )
    if pseudo_is_empty:
        materialization["pseudo"] = create_empty_pseudo_dir(
            output_dir / "pseudo",
            annotation_file_name=str(cfg.pseudo_annotations_file),
        )
    else:
        assert pseudo_dir is not None
        materialization["pseudo"] = materialize_dataset_dir(
            source_dir=pseudo_dir,
            destination_dir=output_dir / "pseudo",
            link_mode=str(cfg.link_mode),
            symlink_fallback_to_copy=bool(cfg.symlink_fallback_to_copy),
        )

    manifest = build_manifest(
        cfg=cfg,
        output_dir=output_dir,
        guardrail_summary=guardrail_summary,
        pseudo_summary=pseudo_summary,
        materialization=materialization,
    )
    write_json(output_dir / "manifest.json", manifest)
    write_round_readme(output_dir, manifest)
    archive_summary = archive_round_dataset(cfg, output_dir, materialization=materialization)
    return summary_from_manifest(manifest, dry_run=False, archive_summary=archive_summary)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="build_round_dataset",
)
def main(cfg: DictConfig) -> None:
    summary = build_round_dataset(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
