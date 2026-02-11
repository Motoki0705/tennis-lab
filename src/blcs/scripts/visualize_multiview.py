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

# Show deprecation warning
warnings.warn(
    "src.blcs.scripts.visualize_multiview is deprecated. "
    "Use 'python -m src.blcs.scripts.visualize' with visualization.cameras=all instead. "
    "This script will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward all arguments to the unified visualize script
if __name__ == "__main__":
    import subprocess
    
    # Build command to call the main visualize script
    args = sys.argv[1:]
    
    # If no config override is specified, use visualize_multiview config
    has_config_override = any(arg.startswith("--config-name") for arg in args)
    if not has_config_override:
        args = ["--config-name=visualize_multiview"] + args
    
    # Call the main script
    cmd = [sys.executable, "-m", "src.blcs.scripts.visualize"] + args
    sys.exit(subprocess.call(cmd))
