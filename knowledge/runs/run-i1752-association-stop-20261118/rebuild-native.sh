#!/usr/bin/env bash
# Preparation added after recording; the original measured command is unchanged.
set -euo pipefail
if [[ ! -e /home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so && ! -e /home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/tennis_lab_compressed_time_local_cuda.so ]]; then
    git submodule update --init third_party/DINO
    CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 bash scripts/build_dino_extension.sh "$PWD" /home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2
fi
.venv/bin/python - /home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib <<'PY'
import hashlib
import sys
from pathlib import Path
folder = Path(sys.argv[1])
expected = {'MultiScaleDeformableAttention.so': {'sha256': '68580f2e5b0454dba630ba3d37e55e383d5babd86b15cda92d4e7c104f5797b7', 'bytes': 12097592}, 'tennis_lab_compressed_time_local_cuda.so': {'sha256': 'b277373744ebad3645cba0cfbf8a41139cb869d3872c2813cbbe12710df8f75e', 'bytes': 13458128}}
if {p.name for p in folder.iterdir()} != set(expected):
    raise ValueError("Native import directory differs from the recorded inventory")
for name, record in expected.items():
    path = folder / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Invalid native binary: {path}")
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != record["sha256"]:
            raise ValueError(f"Native binary checksum differs: {path}")
print("Native import inventory verified")
PY
