set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
export MAX_JOBS=2
mkdir -p outputs/synthetic_data_generation/profile/court-scale-pilot/setup
TIMEFORMAT="elapsed_seconds=%R"
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
time uv sync --locked
time .venv/bin/python -m spin setup-nht --render-only
git submodule update --init third_party/dinov3
nht-render --help
git submodule status third_party/nht third_party/dinov3
git -C third_party/nht/gsplat rev-parse HEAD
.venv/bin/python --version
