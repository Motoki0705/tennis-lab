"""Open a local human-alignment editor against an explicit canonical scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.synthetic_data_generation.alignment.manual.service import AlignmentEditor
from src.synthetic_data_generation.alignment.manual.web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--recover-ground-frame",
        action="store_true",
        help="Explicitly recover a missing frame from verified paired UV/3D observations.",
    )
    args = parser.parse_args()
    editor = AlignmentEditor(
        args.scene_root, recover_ground_frame=args.recover_ground_frame
    )
    print(
        f"Court Alignment Studio · {editor.root.name} · http://localhost:{args.port}",
        flush=True,
    )
    print(f"Ground frame: {editor.source.provenance}", flush=True)
    uvicorn.run(create_app(editor), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
