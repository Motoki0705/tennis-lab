#!/usr/bin/env bash
set -euo pipefail
cd /home/kamimura/.local/state/tennis-lab-agents/campaign-1267/clean-env/standard-install/repo && CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg /home/kamimura/.local/state/tennis-lab-agents/campaign-1267/clean-env/standard-install/repo/.venv/bin/python -m src.tennis_scene.scripts.visualization input=tennis_scene/generate/clean_install/scene.npz output=tennis_scene/visualize/clean_install/scene.mp4 style.player_representation=smpl
