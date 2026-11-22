#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_root=$(realpath -m -- "$HOME/.venvs")
prefix=$(realpath -m -- "${1:-$env_root/tennis-lab-trt}")
case "$prefix" in
    "$env_root"/tennis-lab-trt|"$env_root"/tennis-lab-trt-*) ;;
    *) printf 'Use a dedicated tennis-lab-trt environment in %s\n' "$env_root" >&2; exit 2 ;;
esac

if [[ ! -x "$prefix/bin/python" ]]; then
    mkdir -p -- "$env_root"
    uv venv --python 3.11.15 "$prefix"
fi
"$prefix/bin/python" -I - "$prefix" <<'PY'
import sys
from pathlib import Path

if Path(sys.prefix).resolve() != Path(sys.argv[1]) or sys.base_prefix == sys.prefix:
    raise SystemExit("Expected an isolated TensorRT experiment virtualenv")
if sys.version_info[:3] != (3, 11, 15):
    raise SystemExit("Expected CPython 3.11.15")
if "include-system-site-packages = false" not in (Path(sys.prefix) / "pyvenv.cfg").read_text().lower():
    raise SystemExit("System site packages must be disabled")
PY
uv pip sync --python "$prefix/bin/python" --index-url https://pypi.org/simple "$bundle/requirements.txt"
uv pip check --python "$prefix/bin/python"

repo_url=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
downloads="$prefix/downloads"
mkdir -p -- "$downloads"
unpack() {
    local name=$1 digest=$2 dest=$3 archive
    archive="$downloads/$name"
    if [[ ! -f "$archive" ]]; then
        curl -fSL --retry 3 "$repo_url/$name" -o "$archive.part"
        printf '%s  %s\n' "$digest" "$archive.part" | sha256sum --check --status
        mv -- "$archive.part" "$archive"
    fi
    printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
    dpkg-deb -x "$archive" "$dest"
}

unpack libnvinfer-bin_10.14.1.48-1+cuda13.0_amd64.deb \
    e5dd160772046901a100b7d4cc56f66e1c911a13653d837bb4eb1c9092681ad0 "$prefix/trtexec-dist"
unpack nsight-systems-2026.5.2_2026.5.2.323-1_amd64.deb \
    ff51b505d19de5f5dcdbebb6e6203894e8355c47466385c6967b3e3feafdb678 "$prefix/nsight-dist"

cat > "$prefix/bin/trtexec" <<'TRT'
#!/usr/bin/env bash
set -euo pipefail
prefix=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
site="$prefix/lib/python3.11/site-packages"
library_paths=("$site/tensorrt_libs")
for lib in "$site"/nvidia/*/lib; do
    [[ ! -d "$lib" ]] || library_paths+=("$lib")
done
bundle_paths=$(IFS=:; printf '%s' "${library_paths[*]}")
export LD_LIBRARY_PATH="$bundle_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$prefix/trtexec-dist/usr/src/tensorrt/bin/trtexec" "$@"
TRT
chmod +x "$prefix/bin/trtexec"
"$prefix/nsight-dist/opt/nvidia/nsight-systems/2026.5.2/bin/nsys" --version
