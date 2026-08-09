#!/usr/bin/env bash
# Install the repository-scoped trusted MCP deploy runner as the kamimura user.

set -euo pipefail
umask 077

readonly EXPECTED_USER="kamimura"
readonly REPOSITORY="Motoki0705/tennis-lab"
readonly REPOSITORY_URL="https://github.com/$REPOSITORY"
readonly RUNNER_LABELS="wsl2,tennis-lab,trusted-mcp-deploy"
RUNNER_NAME="$(hostname)-wsl2-trusted-mcp-deploy"
readonly RUNNER_NAME
readonly RUNNER_ROOT="$HOME/.local/share/tennis-lab-trusted-mcp-deploy-runner"
readonly HOOK_ROOT="$HOME/.local/libexec/tennis-lab-actions"
readonly HOOK_PATH="$HOOK_ROOT/authorize_trusted_mcp_deploy_job.sh"
readonly SERVICE_NAME="tennis-lab-trusted-mcp-deploy-runner.service"
readonly SERVICE_DIR="$HOME/.config/systemd/user"
readonly SERVICE_PATH="$SERVICE_DIR/$SERVICE_NAME"
readonly MCP_STATE_DIR="$HOME/.local/state/tennis-lab-chatgpt-mcp"
readonly MCP_CONTROL_DIR="$HOME/.local/share/tennis-lab-chatgpt-mcp"
readonly PROJECT_ROOT="$HOME/projects/tennis-lab"
readonly UV_CACHE_DIR="$HOME/.cache/uv"
readonly UV_INSTALL_DIR="$HOME/.local/share/uv"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_HOOK="$SCRIPT_DIR/authorize_trusted_mcp_deploy_job.sh"
registration_token=""

usage() {
  echo "usage: $0 [--registration-token-stdin]" >&2
}

if [[ "${1:-}" == "--registration-token-stdin" ]]; then
  if [[ $# -ne 1 ]]; then
    usage
    exit 2
  fi
  IFS= read -r registration_token
elif [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

if [[ "$(id -un)" != "$EXPECTED_USER" ]]; then
  echo "Run this installer as $EXPECTED_USER without sudo." >&2
  exit 1
fi
expected_home="$(getent passwd "$EXPECTED_USER" | cut -d: -f6)"
if [[ "$HOME" != "$expected_home" ]]; then
  echo "HOME must be $expected_home, got $HOME." >&2
  exit 1
fi

for command in curl cut getent install jq sha256sum systemctl systemd-analyze tar; do
  if ! command -v "$command" >/dev/null; then
    echo "Required command is unavailable: $command" >&2
    exit 1
  fi
done
for path in "$SOURCE_HOOK" "$MCP_STATE_DIR" "$MCP_CONTROL_DIR" "$PROJECT_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required trusted path is missing: $path" >&2
    exit 1
  fi
done
if [[ "$REPOSITORY_ROOT" != "$PROJECT_ROOT" ]]; then
  echo "Run this installer from the canonical checkout at $PROJECT_ROOT." >&2
  exit 1
fi

install -d -m 0700 \
  "$RUNNER_ROOT" \
  "$HOOK_ROOT" \
  "$SERVICE_DIR" \
  "$UV_CACHE_DIR" \
  "$UV_INSTALL_DIR"
install -m 0500 "$SOURCE_HOOK" "$HOOK_PATH"

download_runner() {
  local temporary_root release_json runner_tag runner_version asset_name url digest
  temporary_root="$(mktemp -d)"
  release_json="$(curl --fail --silent --show-error --location \
    https://api.github.com/repos/actions/runner/releases/latest)"
  runner_tag="$(jq -r .tag_name <<<"$release_json")"
  runner_version="${runner_tag#v}"
  asset_name="actions-runner-linux-x64-${runner_version}.tar.gz"
  url="$(jq -r --arg name "$asset_name" \
    '.assets[] | select(.name == $name) | .browser_download_url' \
    <<<"$release_json")"
  digest="$(jq -r --arg name "$asset_name" \
    '.assets[] | select(.name == $name) | .digest' \
    <<<"$release_json")"
  if [[ -z "$url" || ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    rm -rf "$temporary_root"
    echo "GitHub did not publish a verified $asset_name asset." >&2
    exit 1
  fi
  curl --fail --silent --show-error --location \
    "$url" --output "$temporary_root/$asset_name"
  printf '%s  %s\n' "${digest#sha256:}" "$temporary_root/$asset_name" \
    | sha256sum --check --status
  tar -xzf "$temporary_root/$asset_name" -C "$RUNNER_ROOT"
  rm -rf "$temporary_root"
}

if [[ ! -x "$RUNNER_ROOT/bin/Runner.Listener" ]]; then
  download_runner
fi

if [[ ! -f "$RUNNER_ROOT/.runner" ]]; then
  if [[ -z "$registration_token" && -t 0 ]]; then
    read -r -s -p "GitHub runner registration token (input hidden): " registration_token
    echo
  fi
  if [[ -z "$registration_token" ]]; then
    echo "A GitHub runner registration token is required for first install." >&2
    exit 1
  fi
  (
    cd "$RUNNER_ROOT"
    ./config.sh \
      --unattended \
      --url "$REPOSITORY_URL" \
      --token "$registration_token" \
      --name "$RUNNER_NAME" \
      --labels "$RUNNER_LABELS" \
      --work _work \
      --replace
  )
fi
unset registration_token

runner_env="$RUNNER_ROOT/.env"
runner_env_tmp="$(mktemp "$RUNNER_ROOT/.env.XXXXXX")"
printf 'ACTIONS_RUNNER_HOOK_JOB_STARTED=%s\n' "$HOOK_PATH" >"$runner_env_tmp"
chmod 0600 "$runner_env_tmp"
mv -f "$runner_env_tmp" "$runner_env"

service_candidate_dir="$(mktemp -d "$SERVICE_DIR/.trusted-mcp-units.XXXXXX")"
service_candidate="$service_candidate_dir/$SERVICE_NAME"
cat >"$service_candidate" <<EOF
[Unit]
Description=Trusted GitHub Actions runner for tennis-lab WSL MCP deployment
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$RUNNER_ROOT
Environment="HOME=$HOME"
Environment="PATH=/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib"
Environment="TMPDIR=/tmp"
ExecStart=/bin/bash $RUNNER_ROOT/run.sh
Restart=always
RestartSec=5
TimeoutStopSec=120
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=read-only
ProtectProc=invisible
ProtectSystem=strict
ReadWritePaths=$RUNNER_ROOT
ReadWritePaths=$MCP_STATE_DIR
ReadWritePaths=$MCP_CONTROL_DIR
ReadWritePaths=$PROJECT_ROOT
ReadWritePaths=$SERVICE_DIR
ReadWritePaths=$UV_CACHE_DIR
ReadWritePaths=$UV_INSTALL_DIR
ReadOnlyPaths=$runner_env
ReadOnlyPaths=$HOOK_PATH
RestrictSUIDSGID=true

[Install]
WantedBy=default.target
EOF
chmod 0600 "$service_candidate"
systemd-analyze --user verify "$service_candidate"
mv -f "$service_candidate" "$SERVICE_PATH"
rmdir "$service_candidate_dir"

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user restart "$SERVICE_NAME"
systemctl --user is-active --quiet "$SERVICE_NAME"

echo "Runner:  $RUNNER_NAME"
echo "Labels:  $RUNNER_LABELS"
echo "Service: $SERVICE_NAME"
echo "Hook:    $HOOK_PATH"
