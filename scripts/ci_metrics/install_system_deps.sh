#!/usr/bin/env bash
set -euo pipefail

missing=()
for binary in ffmpeg rclone; do
  if ! command -v "$binary" >/dev/null; then
    missing+=("$binary")
  fi
done

if ((${#missing[@]})); then
  options=(-o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
  timeout 180s sudo apt-get "${options[@]}" update
  timeout 180s sudo apt-get "${options[@]}" install -y --no-install-recommends "${missing[@]}"
fi

# Presence alone is insufficient: failed installs and unusable binaries fail CI.
ffmpeg -version
rclone version
