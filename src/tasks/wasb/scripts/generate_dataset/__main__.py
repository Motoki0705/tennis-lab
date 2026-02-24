"""Default entrypoint for WASB dataset generation (Hydra-based).

Usage:
    `uv run python -m src.tasks.wasb.scripts.generate_dataset`
"""

from __future__ import annotations

from src.tasks.wasb.scripts.generate_dataset.batch import main

if __name__ == "__main__":
    main()

