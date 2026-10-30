set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
OUT=outputs/synthetic_data_generation/profile/court-scale-smoke/setup
mkdir -p "$OUT"
TIMEFORMAT="elapsed_seconds=%R"
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
time uv sync --locked
time MAX_JOBS=2 .venv/bin/python -m spin setup-nht --render-only
nht-render --help
git submodule status third_party/nht
git -C third_party/nht/gsplat rev-parse HEAD
.venv/bin/python --version
