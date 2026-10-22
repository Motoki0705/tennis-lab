#!/usr/bin/env bash
# Common headless runner. Arguments and the local result contract: README.md.
set -euo pipefail
lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$lib_dir/../../../.venv/bin/python" "$lib_dir/agent_runtime.py" "$@"
