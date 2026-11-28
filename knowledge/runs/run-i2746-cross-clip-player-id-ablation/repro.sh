#!/usr/bin/env bash
set -euo pipefail

# Reproducibility script for Issue #2746:
# Cross-clip player identity linking ablation on meiji_4cam/video_000 (multiview-v4, clip_000..068).
#
# Pre-extracted GPU feature caches (created via shared training-queue on CUDA):
#   1) CLIP-ReID cache (4,416 crops across 69 clips):
#      PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/extract_cache.py \
#        --device cuda --batch-size 64
#   2) KPR Market/SOLIDER 6-part cache (unprompted + 2D-keypoint prompted, same 4,416 crops):
#      PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/extract_kpr_cache.py \
#        --device cuda --batch-size 32
#
# Pass --extract-gpu to re-run GPU cache extraction before CPU evaluation (only inside training-queue).

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CACHE_DIR="outputs/i2746_cross_clip_player_id/cache"

if [[ "${1:-}" == "--extract-gpu" ]]; then
  PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/extract_cache.py --device cuda --batch-size 64
  PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/extract_kpr_cache.py --device cuda --batch-size 32
fi

if [[ ! -f "$CACHE_DIR/clip_reid_cache.npz" ]]; then
  echo "ERROR: missing $CACHE_DIR/clip_reid_cache.npz (extract via training-queue first)" >&2
  exit 1
fi

echo "=== 1. Contract & metric verification tests ==="
PYTHONPATH=. .venv/bin/python -m pytest -q knowledge/runs/run-i2746-cross-clip-player-id-ablation/test_contract.py

echo "=== 2. E0..E7 isolated ablation (dev calibration + frozen test evaluation) ==="
PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/run_e0_e7.py --eval-test

echo "=== 3. Follow-up axes (LLR, Swap-Margin, Axis G greedy/MILP, Axis P KPR) ==="
PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/followup.py --eval-test

echo "=== 4. Export compact CSV/JSON artifacts & versioned player_identity_links sample ==="
PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/export_artifacts.py

echo "=== 5. Generate ablation comparison figure & 4-view identity montage ==="
PYTHONPATH=. .venv/bin/python knowledge/runs/run-i2746-cross-clip-player-id-ablation/make_figures.py

echo "=== Done: knowledge/runs/run-i2746-cross-clip-player-id-ablation ==="
