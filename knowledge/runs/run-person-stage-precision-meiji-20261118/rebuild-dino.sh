#!/usr/bin/env bash
# Replay the measured DINO build on CPU. Stdout returns the verified binary path.
set -euo pipefail

fail() {
    printf '[rebuild-dino] %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 2 ]] || fail "usage: $0 <input-pin.json> <build-directory>"
repo="$(git rev-parse --show-toplevel)"
python="$repo/.venv/bin/python"
build_commit=5854fa629ac8891ae9210d553c72e3c9ca5a9213
cuda_home=/usr/local/cuda-13.0
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME CFLAGS CPPFLAGS CXXFLAGS LDFLAGS CC CXX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

metadata="$(CUDA_VISIBLE_DEVICES='' "$python" - "$1" <<'PY'
import importlib.metadata
import json
import re
import sys
import sysconfig
from pathlib import Path

import torch

expected = ("3.11.15", "2.13.0+cu130", "13.0", "81.0.0")
actual = (".".join(map(str, sys.version_info[:3])), torch.__version__, torch.version.cuda,
          importlib.metadata.version("setuptools"))
if actual != expected:
    raise SystemExit(f"Expected Python, torch, torch CUDA and setuptools {expected}, found {actual}")
pin = json.loads(Path(sys.argv[1]).read_text())
for name, length in (("dino_binary_sha256", 64), ("dino_submodule_commit", 40)):
    value = pin[name]
    if not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise SystemExit(f"Invalid {name} in input pin")
    print(value)
print(Path(sysconfig.get_path("purelib")).resolve(strict=True))
print(Path(sys.base_prefix).resolve(strict=True))
PY
)"
mapfile -t native < <(printf '%s\n' "$metadata")
expected_sha="${native[0]}"
dino_commit="${native[1]}"
mkdir -p "$2/lib"
output="$(cd -- "$2" && pwd -P)"
binary="$output/lib/MultiScaleDeformableAttention.so"
exec 9>"$output/.build.lock"
flock 9

verify_binary() {
    [[ -f "$1" && ! -L "$1" ]] || fail "expected a regular DINO binary at $1"
    local actual
    actual="$(sha256sum < "$1")"
    [[ "${actual%% *}" == "$expected_sha" ]] ||
        fail "DINO SHA-256 mismatch at $1 (expected $expected_sha, got ${actual%% *})"
}

if [[ -e "$binary" || -L "$binary" ]]; then
    verify_binary "$binary"
    printf '%s\n' "$binary"
    exit 0
fi

require_version() {
    local expected="$1" actual
    shift
    actual="$("$@")"
    [[ "$actual" == *"$expected"* ]] || fail "expected $expected from $*, got $actual"
}

require_version 'V13.0.88' "$cuda_home/bin/nvcc" --version
require_version 'compiler.36424714_0' "$cuda_home/bin/nvcc" --version
require_version 'gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1)' gcc --version
require_version 'c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1)' c++ --version
require_version 'GNU ld (GNU Binutils for Ubuntu) 2.42' ld --version
require_version '1.11.1' ninja --version
[[ "$(git -C "$repo" rev-parse "$build_commit:third_party/DINO")" == "$dino_commit" ]] ||
    fail "the DINO input pin disagrees with the build revision"
if ! git -C "$repo/third_party/DINO" cat-file -e "$dino_commit^{commit}" 2>/dev/null; then
    git -C "$repo" submodule update --init --checkout -- third_party/DINO >&2
fi

stage="$(mktemp -d /tmp/tennis-dino-repro.XXXXXX)"
trap 'rm -rf -- "$stage"' EXIT
# GCC and NVCC encode these paths in the measured ELF. A private mount
# namespace supplies all of them from the staged copies and read-only venv.
recorded_assets=/home/kamimura/projects/tennis-lab
recorded_repo="$recorded_assets/.claude/worktrees/video-batch-cli"
recorded_outputs="$recorded_assets/.worktree-outputs/video-batch-cli"
build_relative=tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2
recorded_python=/home/kamimura/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu
staged_home="$stage/home"
staged_assets="$staged_home/${recorded_assets#/home/}"
staged_repo="$staged_home/${recorded_repo#/home/}"
staged_build="$staged_home/${recorded_outputs#/home/}/$build_relative"
staged_python="$staged_home/${recorded_python#/home/}"
staged_packages="$staged_repo/.venv/lib/python3.11/site-packages"
mkdir -p "$staged_build" "$staged_python" "$staged_packages" \
    "$staged_assets/"{third_party/DINO,data,ckpt} "$staged_repo/.venv/"{bin,include}
git -C "$repo" archive "$build_commit" \
    setup.py pyproject.toml src README.md LICENSE NOTICE LICENSES.md LICENSES.ja.md scripts/build_dino_extension.sh |
    tar -x -C "$staged_repo"
git -C "$repo/third_party/DINO" archive "$dino_commit" models/dino/ops/src |
    tar -x -C "$staged_assets/third_party/DINO"
git init -q "$staged_repo"
ln -s "$recorded_outputs" "$staged_repo/outputs"
ln -s "$recorded_python/bin/python3.11" "$staged_repo/.venv/bin/python"
printf 'home = %s/bin\ninclude-system-site-packages = false\nversion = 3.11.15\n' "$recorded_python" \
    > "$staged_repo/.venv/pyvenv.cfg"

cat > "$stage/nvcc" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == */dino_ops_sources/cuda/ms_deform_attn_cuda.cu ]]; then
        # The host stub's ELF file symbol contains the NVCC PID (000fcefb).
        exec unshare --mount --pid --fork --mount-proc --propagation private bash -c '
            set -euo pipefail
            test "$$" = 1
            printf "%s\n" "$((0xfcefb - 1))" >/proc/sys/kernel/ns_last_pid
            "$@" &
            child=$!
            test "$child" -eq "$((0xfcefb))"
            wait "$child"
        ' nvcc "$CUDA_HOME/bin/nvcc" "$@"
    fi
done
exec "$CUDA_HOME/bin/nvcc" "$@"
SH
chmod +x "$stage/nvcc"

printf '[rebuild-dino] compiling with CUDA hidden. Log %s\n' "$output/build.log" >&2
if ! env CUDA_HOME="$cuda_home" CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
    PYTORCH_NVCC="$stage/nvcc" MPLCONFIGDIR="$stage/matplotlib" \
    unshare --user --map-root-user --mount --propagation private bash -c '
        set -euo pipefail
        mount --bind "$1" "$2"
        mount -o remount,bind,ro "$2"
        mount --bind "$3" "$4"
        mount -o remount,bind,ro "$4"
        mount --rbind "$5" /home
        cd "$6"
        bash scripts/build_dino_extension.sh "$7" "$8"
    ' native-build "${native[2]}" "$staged_packages" "${native[3]}" "$staged_python" "$staged_home" \
        "$recorded_repo" "$recorded_assets" "$recorded_repo/outputs/$build_relative" > "$output/build.log" 2>&1; then
    fail "CPU build failed. See $output/build.log"
fi
rebuilt="$staged_build/lib/MultiScaleDeformableAttention.so"
verify_binary "$rebuilt"
cp -- "$staged_build/build.json" "$output/build.json"
cp -- "$rebuilt" "$output/lib/.MultiScaleDeformableAttention.partial"
mv -- "$output/lib/.MultiScaleDeformableAttention.partial" "$binary"
printf '%s\n' "$binary"
