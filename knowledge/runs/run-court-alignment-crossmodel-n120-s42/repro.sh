#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${run_dir}/../../.." && pwd)"
ours_checkpoint="${COURT_CHECKPOINT:-${repo_root}/outputs/court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt}"
tcd_repo="${TCD_REPO:?Set TCD_REPO to yastrebksv/TennisCourtDetector at commit e5cd4f1ce26b15361700d3d89e068cbf0e82749e}"
tcd_checkpoint="${TCD_CHECKPOINT:?Set TCD_CHECKPOINT to model_best.pt with SHA-256 09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04}"

cd "${repo_root}"
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m src.tasks.court_detection.scripts.benchmark_alignment \
  --output-dir "${run_dir}" \
  --repo-root "${repo_root}" \
  --datasets all \
  --models all \
  --max-samples-per-domain 120 \
  --seed 42 \
  --ours-checkpoint "${ours_checkpoint}" \
  --tcd-repo "${tcd_repo}" \
  --tcd-checkpoint "${tcd_checkpoint}"
