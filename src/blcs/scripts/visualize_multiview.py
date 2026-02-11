"""DEPRECATED: Use src.blcs.scripts.visualize instead.

This script is deprecated and maintained only for backward compatibility.
All multi-view visualization functionality has been integrated into the
unified visualize.py entry point.

Please use:
    python -m src.blcs.scripts.visualize visualization.cameras=all

Or with config override:
    python -m src.blcs.scripts.visualize --config-name=visualize_multiview
"""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "src.blcs.scripts.visualize_multiview is deprecated. "
    "Use 'python -m src.blcs.scripts.visualize' with visualization.cameras=all instead. "
    "This script will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the main entry point for backward compatibility
from src.blcs.scripts.visualize import main

if __name__ == "__main__":
    sys.exit(main())
