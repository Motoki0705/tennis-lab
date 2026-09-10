#!/usr/bin/env bash
set -euo pipefail
# Run from the matching repository revision; data/assets must be provisioned.
# Full observed configuration is config.yaml beside this script.
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_axial_reference "$@"
