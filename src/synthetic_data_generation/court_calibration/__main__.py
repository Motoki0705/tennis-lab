"""Small CPU CLI using existing OmegaConf output and typed path resolvers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
from omegaconf import OmegaConf

from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

from .database import DatabaseConfig, LineDatabase, generate
from .matching import query_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/baseline.yaml"
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
    if output.exists():
        raise FileExistsError(output)
    db_path = resolver.resolve(PathRole.DATA, config.database_path)
    if config.mode == "generate":
        database = generate(db_config)
        # Exclusive DB publication rejects accidental reuse/overwrite.
        database.save(db_path)
        report = {
            "database": str(db_path),
            "count": db_config.count,
            "metadata": db_config.metadata(),
        }
    elif config.mode == "query":
        database = LineDatabase.load(db_path, expected_config=db_config)
        query_path = resolver.resolve(PathRole.DATA, config.query_path)
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
