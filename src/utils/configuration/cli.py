"""Explicit CLI selection; leaf commands retain their schema and path validators."""
from collections.abc import Mapping


def dispatch_cli(selector: str, entries: Mapping[str, str], *, description: str) -> None:
    """Parse an explicit operation and pass remaining options to its boundary."""
    import argparse
    import importlib
    import sys

    flag = f"--{selector}"
    parser = argparse.ArgumentParser(description=description, add_help=not any(
        arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]))
    parser.add_argument(flag, choices=tuple(entries), required=True)
    selected, remaining = parser.parse_known_args()
    module = importlib.import_module(entries[getattr(selected, selector)])
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        module.main()
    finally:
        sys.argv = previous
