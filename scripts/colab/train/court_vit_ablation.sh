#!/usr/bin/env bash
# The workflow owns L4 provisioning, input staging and Drive credentials.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
sha256sum --check --strict scripts/colab/train/court_vit_ablation/baseline_checkpoint.sha256
if ! command -v zstd >/dev/null; then
    apt-get update
    apt-get install -y zstd
fi
.venv/bin/python -m scripts.colab.train.court_vit_ablation.prepare
exec .venv/bin/python -m scripts.colab.train.court_vit_ablation.run "$@"
