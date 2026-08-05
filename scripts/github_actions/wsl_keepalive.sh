#!/usr/bin/env bash
# Keep a Windows-side wsl.exe handle open so WSL services remain reachable.

set -euo pipefail

while true; do
  sleep 2147483647 &
  wait "$!"
done
