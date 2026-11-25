#!/usr/bin/env bash
set -euo pipefail

pmpose_site=$(realpath -m "${1:?Supply a private directory for PMPose dependencies}")
pmpose_run=$(dirname "${BASH_SOURCE[0]}")
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$pmpose_site"
# MMPretrain 1.2.0 imports legacy Transformers BERT utilities when registering
# its models. Install the compatible version in this private directory.
uv pip install --python .venv/bin/python --target "$pmpose_site" --link-mode=copy \
    --no-deps -r "$pmpose_run/pmpose-requirements.txt"

# The published xtcocotools wheel is linked to NumPy 1.x. Build against the
# NumPy already used by the common benchmark environment.
uv --no-cache pip install --python .venv/bin/python --target "$pmpose_site" \
    --link-mode=copy --no-deps --no-build-isolation --no-binary xtcocotools \
    --reinstall-package xtcocotools xtcocotools==1.14.3

# Other upstream MMPose heads import mmcv.ops at registration time. PMPose's
# forward uses Torch modules and does not use these compiled operators.
MMCV_WITH_OPS=1 MAX_JOBS=3 CFLAGS=-O1 CXXFLAGS=-O1 \
    uv pip install --python .venv/bin/python --target "$pmpose_site" --link-mode=copy \
    --no-deps --no-build-isolation --no-binary mmcv mmcv==2.1.0

.venv/bin/python - <<'PY'
import torch
import mmcv.ops
import xtcocotools.mask
from mmpretrain.models.backbones import VisionTransformer

print(torch.__version__, mmcv.__version__, VisionTransformer.__name__)
PY
