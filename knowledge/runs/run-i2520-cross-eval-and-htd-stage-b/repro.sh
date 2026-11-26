#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/eval_results}"
mkdir -p "$OUT_DIR"

export PYTHONPATH=.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_b0_multiview_axial_e100_s42/logs/version_0/checkpoints/plcs-epoch=99.ckpt \
  --device cpu --mode synthetic --output-json "$OUT_DIR/B0.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_m0_mono_gvhmr_base_e100_s42/logs/version_0/checkpoints/plcs-epoch=64.ckpt \
  --device cpu --mode all --output-json "$OUT_DIR/M0.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_m1_mono_gvhmr_htd_e100_s42/logs/version_0/checkpoints/plcs-epoch=59.ckpt \
  --device cpu --mode all --output-json "$OUT_DIR/M1.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_m1_mono_gvhmr_htd_e100_s42/logs/version_0/checkpoints/plcs-epoch=59.ckpt \
  --device cpu --mode all --htd-refine --htd-refine-mode single_view --htd-steps 25 \
  --output-json "$OUT_DIR/M2_SV.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_m1_mono_gvhmr_htd_e100_s42/logs/version_0/checkpoints/plcs-epoch=59.ckpt \
  --device cpu --mode meiji --htd-refine-mode multiview --htd-steps 25 \
  --output-json "$OUT_DIR/M2_MV.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_a1_no_cal_noise_e100_s42/logs/version_0/checkpoints/last.ckpt \
  --device cpu --mode all --output-json "$OUT_DIR/A1.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_a2_no_bbox_aug_e100_s42/logs/version_0/checkpoints/plcs-epoch=59.ckpt \
  --device cpu --mode all --output-json "$OUT_DIR/A2.json"

.venv/bin/python "$SCRIPT_DIR/eval_i2520_plcs.py" \
  --checkpoint outputs/plcs/i2520_v2_a3_world3d_only_e100_s42/logs/version_0/checkpoints/last.ckpt \
  --device cpu --mode all --output-json "$OUT_DIR/A3.json"
