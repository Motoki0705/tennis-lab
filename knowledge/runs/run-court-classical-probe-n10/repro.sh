#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${bundle_dir}/../../.." && pwd)"
work_root="$(mktemp -d -t tennis-court-classical.XXXXXXXX)"
upstream_commit="762d077541a77abf4923f5f8f689a1410927d35e"

mkdir -p "${work_root}/patches" "${work_root}/scripts"
cp "${bundle_dir}/Dockerfile" "${bundle_dir}/probe.cpp" "${work_root}/"
cp "${bundle_dir}/tcd_opencv4.patch" "${work_root}/patches/"
cp "${bundle_dir}/scripts/run_sweep.py" "${work_root}/scripts/"

git clone https://github.com/gchlebus/tennis-court-detection.git \
  "${work_root}/tcd-src"
git -C "${work_root}/tcd-src" checkout --quiet "${upstream_commit}"
git -C "${work_root}/tcd-src" apply "${work_root}/patches/tcd_opencv4.patch"

# Ubuntu 22.04 とディストリビューション版 OpenCV 4.5.4 のみを使う。
# --gpus は指定せず、実行時のコンテナネットワークも無効にする。
docker build -t tcd-opencv4:local "${work_root}"

TENNIS_LAB_REPO_ROOT="${repo_root}" \
TCD_WORK_DIR="${work_root}" \
  "${repo_root}/.venv/bin/python" "${work_root}/scripts/run_sweep.py"

echo "classical probe artifacts: ${work_root}/extra"
