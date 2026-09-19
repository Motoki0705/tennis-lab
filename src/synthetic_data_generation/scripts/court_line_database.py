"""Small CPU CLI using existing OmegaConf output and typed path resolvers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
from omegaconf import OmegaConf

from src.synthetic_data_generation.court_calibration.database import (
    DatabaseConfig,
    LineDatabase,
    generate,
)
from src.synthetic_data_generation.court_calibration.matching import query_database
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court_line_database",
    fields=(
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)
GENERATE_PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court_line_database",
    fields=(
        BoundaryPathField(
            "database", PathRole.DATA, PathDirection.OUTPUT, PathKind.FILE
        ),
    ),
)
QUERY_PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court_line_database",
    fields=(
        BoundaryPathField(
            "database",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "query", PathRole.DATA, PathDirection.INPUT, PathKind.FILE, must_exist=True
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "src/synthetic_data_generation/court_calibration/configs/baseline.yaml",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict) or set(raw) != {
        "mode",
        "paths",
        "output_dir",
        "database_path",
        "query_path",
        "database",
        "matching",
    }:
        raise ValueError("CLI configuration keys do not match")
    db_config = DatabaseConfig.from_mapping(raw["database"])
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(dict(config.paths), repository_root=PROJECT_ROOT)
    )
    output = resolver.resolve(PathRole.OUTPUT, config.output_dir)
    PATH_BOUNDARY.validate({"output": output}, resolver=resolver)
    if output.exists():
        raise FileExistsError(output)
    db_path = resolver.resolve(PathRole.DATA, config.database_path)
    if config.mode == "generate":
        GENERATE_PATH_BOUNDARY.validate({"database": db_path}, resolver=resolver)
        database = generate(db_config)
        # Exclusive DB publication rejects accidental reuse/overwrite.
        database.save(db_path)
        report = {
            "database": str(db_path),
            "count": db_config.count,
            "metadata": db_config.metadata(),
        }
    elif config.mode == "query":
        query_path = resolver.resolve(PathRole.DATA, config.query_path)
        QUERY_PATH_BOUNDARY.validate(
            {"database": db_path, "query": query_path}, resolver=resolver
        )
        database = LineDatabase.load(db_path, expected_config=db_config)
        mask = cv2.imread(str(query_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"cannot read query mask: {query_path}")
        matches = query_database(database, mask, **dict(config.matching))
        report = {
            "database": str(db_path),
            "query": str(query_path),
            "matches": [
                {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in asdict(match).items()
                }
                for match in matches
            ],
        }
    else:
        raise ValueError("mode must be generate or query")
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    (output / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(output / "result.json")


if __name__ == "__main__":
    main()
