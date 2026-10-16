#!/usr/bin/env bash
set -euo pipefail
export UV_TOOL_DIR="${TENNIS_COLAB_TOOLS_DIR:-$HOME/.local/share/tennis-lab/colab-tools}"
export UV_TOOL_BIN_DIR="$UV_TOOL_DIR/bin"
uv tool install --python 3.12 google-colab-cli==0.7.4
