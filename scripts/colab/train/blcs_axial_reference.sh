#!/usr/bin/env bash
# Train the fixed four-corner BLCS reference recipe on Colab VM local disk.
# Drive mounting and environment setup belong to the Colab workflow runner.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
sha256sum --check --strict scripts/colab/train/blcs_axial_reference.dataset.sha256
if [[ -L data/blcs/single_object_camera_view_v2 ]]; then
    echo "Refusing a symlink dataset destination." >&2
    exit 2
fi
# A retry restores the same verified immutable dataset before training.
tar --extract --gzip --file data/blcs/single_object_camera_view_v2.tar.gz --directory data/blcs
exec .venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_axial_reference "$@"
