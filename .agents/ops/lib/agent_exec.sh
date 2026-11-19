#!/usr/bin/env bash
# Common headless runner. Arguments and the local result contract: README.md.
# Bash parses this whole command before running it. It never rereads the file
# while Python waits for the checkout lock before importing the runtime.
{
    set -euo pipefail
    lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    exec "$lib_dir/../../../.venv/bin/python" "$lib_dir/agent_runtime.py" "$@"
}
