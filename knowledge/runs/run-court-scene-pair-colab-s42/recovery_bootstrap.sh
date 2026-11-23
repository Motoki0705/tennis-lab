#!/usr/bin/env bash
set -euo pipefail
cd /content/tennis-lab
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
: "${RCLONE_CONFIG:?Authenticated session configuration required}"
: "${COURT_RECOVERY_DRIVE_CLI:?Versioned current Drive CLI required}"
: "${COURT_RECOVERY_ADAPTER_SHA256:?Exact operational adapter digest required}"
: "${COURT_RECOVERY_OPERATIONAL_REF:?Exact operational commit required}"
uv sync --locked --python 3.11
git submodule update --init third_party/dinov3
.venv/bin/python - <<'PY'
import sys
import torch
import torchvision
if (sys.version_info[:2] != (3, 11) or torch.__version__ != "2.14.1+cu130"
        or torchvision.__version__ != "0.29.1+cu130"):
    raise RuntimeError("Pinned Court recovery Python/Torch/torchvision identity changed")
PY
exec .venv/bin/python knowledge/runs/run-court-scene-pair-colab-s42/recovery_adapter.py run \
  --drive-cli "$COURT_RECOVERY_DRIVE_CLI"
