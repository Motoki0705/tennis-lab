#!/usr/bin/env bash
# Reproducible end-to-end run of the classical tennis-court baseline
# (gchlebus/tennis-court-detection @ 762d077541a77abf4923f5f8f689a1410927d35e) on CPU.
#
# Everything happens under /tmp; the tennis-lab repo is only read.
# Prerequisites on the host: git, docker (29.3.1 tested), ffmpeg 6.1.1,
# and /home/kamimura/projects/tennis-lab/.venv/bin/python for the synthetic RGB decode.
set -euo pipefail

WORK=/tmp/tcd_work
UPSTREAM_CLONE=/tmp/field_align_scout/tcd1      # local clone of the upstream repository
COMMIT=762d077541a77abf4923f5f8f689a1410927d35e

# 1. pristine checkout at the fixed commit + minimal OpenCV-4 patch
if [ ! -d "${WORK}/tcd-src" ]; then
  git clone --no-hardlinks "${UPSTREAM_CLONE}" "${WORK}/tcd-src"
fi
git -C "${WORK}/tcd-src" checkout --quiet "${COMMIT}"
git -C "${WORK}/tcd-src" apply "${WORK}/patches/tcd_opencv4.patch"

# 2. CPU-only container image (Ubuntu 22.04 + distro OpenCV 4.5.4, no CUDA)
docker build -t tcd-opencv4:local "${WORK}"

# 3. inputs
#    real photo  -> 3-frame lossless FFV1 AVI (algorithm reads the middle frame)
/home/kamimura/projects/tennis-lab/.venv/bin/python "${WORK}/scripts/make_synth_png.py" > "${WORK}/out/synth_png_info.json"
ffmpeg -hide_banner -loglevel error -y -loop 1 -i "${WORK}/inputs/real_EF-hx40Q4Mg_700.png" \
  -frames:v 3 -c:v ffv1 -pix_fmt bgr0 -f avi "${WORK}/inputs/real_EF-hx40Q4Mg_700.ffv1.avi"
ffmpeg -hide_banner -loglevel error -y -loop 1 -i "${WORK}/inputs/synth_court-sample-000897.png" \
  -frames:v 3 -c:v ffv1 -pix_fmt bgr0 -f avi "${WORK}/inputs/synth_court-sample-000897.ffv1.avi"

# 4. runs (usage / real / synthetic), bounded by /usr/bin/time -v for RSS + elapsed
docker run --rm --network none tcd-opencv4:local /build/build/detect    # usage -> exit 255
docker run --rm --network none -v "${WORK}/out:/out" -v "${WORK}/inputs:/in" \
  -e TCD_OVERLAY=/out/real_overlay.png tcd-opencv4:local \
  /usr/bin/time -v /build/build/detect /in/real_EF-hx40Q4Mg_700.ffv1.avi /out/real_result.txt
docker run --rm --network none -v "${WORK}/out:/out" -v "${WORK}/inputs:/in" \
  -e TCD_OVERLAY=/out/synth_primary_overlay.png tcd-opencv4:local \
  /usr/bin/time -v /build/build/detect /in/synth_court-sample-000897.ffv1.avi /out/synth_primary_result.txt
