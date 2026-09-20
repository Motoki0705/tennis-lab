#!/usr/bin/env bash
# Download + CPU verification only. Never replace the installed checkpoint automatically.
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
run_dir="${1:?Usage: bash repro.sh NEW_OUTPUT_DIRECTORY}"
mkdir "$run_dir"
curl --fail --location --retry 2 --connect-timeout 30 --max-time 1200 \
  --silent --show-error --dump-header "$run_dir/http_headers.txt" \
  --output "$run_dir/vitpose-h-multi-coco.pth.part" \
  --write-out '%{http_code}\n%{size_download}\n' \
  'https://huggingface.co/camenduru/GVHMR/resolve/d50e513e101f7465d15508dc4c797c328200c78e/vitpose/vitpose-h-multi-coco.pth' \
  > "$run_dir/curl_result.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$repo_root/.venv/bin/python" "$repo_root/knowledge/runs/run-slcs-vitpose-redownload-v1/verify.py" \
  "$run_dir/vitpose-h-multi-coco.pth.part" > "$run_dir/verification.log" 2>&1
