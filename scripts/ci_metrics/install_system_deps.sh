#!/usr/bin/env bash
set -euo pipefail

# All bytes, versions and signatures are pinned; no sudo/apt or PATH fallback.
exec python3 -m scripts.ci_metrics.system_deps "$@"
