"""Stage captured VM control helpers before replaying the unchanged Colab request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(destination: Path) -> None:
    bundle = Path(__file__).parent
    manifest = json.loads((bundle / "control-sources.json").read_text())
    for relative, expected in manifest["files"].items():
        source = bundle / "control-source" / relative
        data = source.read_bytes()
        if (
            len(data) != expected["bytes"]
            or hashlib.sha256(data).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Captured control source changed: {relative}")
        target = destination / relative
        if target.exists() and target.read_bytes() != data:
            raise ValueError(
                f"Refusing a different existing control helper: {relative}"
            )
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes((bundle / "control-source" / relative).read_bytes())
    historical = (bundle / "colab-repro.sh").read_bytes()
    if (
        hashlib.sha256(historical).hexdigest()
        != manifest["generated_colab_repro_sha256"]
    ):
        raise ValueError("Generated historical Colab replay changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    prepare(parser.parse_args().destination)
