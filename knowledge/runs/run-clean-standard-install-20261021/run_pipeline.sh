#!/usr/bin/env bash
set -euo pipefail
cd /home/kamimura/.local/state/tennis-lab-agents/campaign-1267/clean-env/standard-install/repo && PYTHONPATH=/home/kamimura/.local/state/tennis-lab-agents/campaign-1267/clean-env/standard-install/repo/outputs/dino_extension/lib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/kamimura/.local/state/tennis-lab-agents/campaign-1267/clean-env/standard-install/repo/.venv/bin/python -m src.tennis_scene.scripts.run_pipeline output_directory=tennis_scene/generate/clean_install output_name=scene cache.directory=tennis_scene/cache/clean_install
