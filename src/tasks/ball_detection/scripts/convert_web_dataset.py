"""Convert supported raw web datasets into the unified ball-detection store.

Each raw layout is isolated behind a dataset-specific parser. This script only
selects parsers, streams their normalized frame records, and persists them via
the web data-access layer.

Usage:
    python -m src.tasks.ball_detection.scripts.convert_web_dataset
    python -m src.tasks.ball_detection.scripts.convert_web_dataset \
        convert.limit_per_source=50 convert.overwrite=true

Notes:
    - Hydra config: ``src/tasks/ball_detection/configs/convert_web_dataset.yaml``.
    - Unknown annotation states are excluded by the source parsers.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from src.tasks.ball_detection.data.components.web.data_access_layer.web_store import (
    INDEX_FILE,
    SCHEMA_VERSION,
    SHARDS_DIR,
    STRINGS_FILE,
)
from src.tasks.ball_detection.data.components.web.data_access_layer.writer import (
    IndexBuilder,
    ShardWriter,
    publish_store,
    write_manifest,
    write_store_readme,
)
from src.tasks.ball_detection.data.components.web.parser import (
    BallYoloParser,
    KaggleParser,
    RacketVisionParser,
    RoboflowParser,
    WebDatasetParser,
)
from src.utils.data.splits import GroupSplitConfig
from src.utils.io import ensure_dir, load_json


def _hydra_main(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return cast(Callable[[Any], Any], hydra.main(*args, **kwargs))


@_hydra_main(
    config_path="../configs",
    config_name="convert_web_dataset",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Run the configured raw-to-unified conversion."""
    convert = cfg.convert
    web_root = Path(to_absolute_path(str(convert.web_root)))
    output_dir = Path(to_absolute_path(str(convert.output_dir)))
    index_path = output_dir / INDEX_FILE

    if index_path.exists() and not bool(convert.overwrite):
        _validate_existing_schema(output_dir)
        print(f"[convert_web_dataset] index exists, skipping: {index_path}")
        print("  pass convert.overwrite=true to rebuild.")
        return 0

    build_dir = output_dir.with_name(f".{output_dir.name}.building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    ensure_dir(build_dir)

    writer = ShardWriter(
        build_dir / SHARDS_DIR,
        int(convert.shard_size_bytes),
    )
    index = IndexBuilder()
    limit = int(convert.limit_per_source)
    parsers = _build_parsers(convert, web_root)

    try:
        for parser in parsers:
            for source in parser.sources():
                added = 0
                for record in tqdm(
                    source.records(),
                    desc=f"convert:{source.name}",
                    unit="frame",
                ):
                    index.add(record, writer)
                    added += 1
                    if limit and added >= limit:
                        break
                print(f"[convert_web_dataset] {source.name}: {added} frames")

        writer.close()
        index.save(build_dir)
        write_manifest(
            build_dir,
            index,
            writer,
            max_bbox_side_ratio=_optional_float(convert.get("max_bbox_side_ratio")),
        )
        write_store_readme(build_dir)
        publish_store(build_dir, output_dir)
    except BaseException:
        writer.close()
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    print(
        f"[convert_web_dataset] wrote {len(index)} samples, "
        f"{writer.shard_count} shard(s), "
        f"{writer.total_bytes / 1e9:.2f} GB packed -> {output_dir}"
    )
    return 0


def _validate_existing_schema(output_dir: Path) -> None:
    strings_path = output_dir / STRINGS_FILE
    existing_schema = None
    if strings_path.exists():
        existing_schema = load_json(strings_path).get("schema")
    if existing_schema != SCHEMA_VERSION:
        raise RuntimeError(
            f"Existing web store schema is {existing_schema!r}, expected "
            f"{SCHEMA_VERSION!r}. Rebuild with convert.overwrite=true."
        )


def _build_parsers(convert: Any, web_root: Path) -> list[WebDatasetParser]:
    split_config = GroupSplitConfig(
        val_ratio=float(convert.val_ratio),
        test_ratio=float(convert.test_ratio),
        seed=int(convert.split_seed),
    )
    quality = int(convert.jpeg_quality)
    max_ratio = _optional_float(convert.get("max_bbox_side_ratio"))
    parsers: list[WebDatasetParser] = []
    if bool(convert.sources.roboflow):
        parsers.append(RoboflowParser(web_root, split_config, max_ratio))
    if bool(convert.sources.racketvision):
        parsers.append(RacketVisionParser(web_root, quality))
    if bool(convert.sources.kaggle):
        parsers.append(
            KaggleParser(
                web_root,
                quality,
                split_config,
                float(convert.kaggle_corner_frac),
            )
        )
    if bool(convert.sources.ball_yolo):
        parsers.append(BallYoloParser(web_root, quality, split_config, max_ratio))
    return parsers


def _optional_float(value: Any) -> float | None:
    """Coerce a possibly-null Hydra scalar to ``float | None``."""
    return None if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
