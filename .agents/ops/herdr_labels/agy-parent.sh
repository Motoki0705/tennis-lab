#!/bin/sh
set -eu

case ${1-} in
    --hook=pre-invocation|--hook=status-line)
        if [ "${TENNIS_LAB_AGY_PARENT:-}" != 1 ]; then
            if [ "$1" = --hook=pre-invocation ]; then
                printf '%s\n' '{}'
            fi
            exit 0
        fi
        ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
python="$repo/.venv/bin/python"
if [ ! -x "$python" ]; then
    printf '%s\n' 'agy parent: .venv/bin/python is required in this checkout; set up its virtual environment' >&2
    exit 1
fi

case ${1-} in
    --hook=pre-invocation|--hook=status-line)
        exec "$python" -I "$script_dir/agy_hook.py" "${1#--hook=}"
        ;;
    *)
        cd "$repo"
        export PYTHONPATH="$repo/.agents/ops"
        exec "$python" -m herdr_labels.agy_tui "$@"
        ;;
esac
