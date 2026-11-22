"""Run the #2305 GVHMR runtime on a plan built from an additive-mask graph.

The #2305 runtime checks layer precision with the shared verifier, which only
recognises Bool attention masks.  This entry point requires the conversion
certificate to name an additive-mask rewrite, then uses
``verify_additive_attention`` for that check.  Everything else is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
for dependency in (
    HERE.parents[2],
    RUNS / "run-i2297-clipreid-tensorrt",
    RUNS / "run-i2305-gvhmr-tensorrt",
    HERE,
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import run_gvhmr  # noqa: E402
from additive_precision import verify_additive_attention  # noqa: E402

ADDITIVE_MODES = ("additive", "additive-finite")


def require_additive(argv: list[str]) -> str:
    if argv.count("--conversion") != 1:
        raise ValueError("The additive runtime requires one --conversion")
    conversion = Path(argv[argv.index("--conversion") + 1])
    certificate = json.loads((conversion / "conversion.json").read_text())
    mode = certificate["graph_rewrite"]["attention_mask"]["mode"]
    if mode not in ADDITIVE_MODES:
        raise ValueError(f"Conversion uses attention mask {mode!r}, not additive")
    return mode


def main() -> None:
    require_additive(sys.argv[1:])
    run_gvhmr.verify_compute_precision = verify_additive_attention
    run_gvhmr.main()


if __name__ == "__main__":
    main()
