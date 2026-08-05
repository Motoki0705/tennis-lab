#!/usr/bin/env bash
# Install the repository-scoped GitHub Actions runner and GPU queue services.

set -euo pipefail

readonly RUNNER_USER="tennis-actions"
readonly RUNNER_GROUP="tennis-actions"
readonly RUNNER_ROOT="/opt/actions-runner"
readonly TOOL_ROOT="/opt/tennis-lab-actions"
readonly STATE_ROOT="/var/lib/tennis-lab-actions"
readonly RUNNER_HOME="$STATE_ROOT/home"
readonly REPOSITORY_URL="https://github.com/Motoki0705/tennis-lab"
readonly RUNNER_LABELS="local-gpu,cuda,wsl2,tennis-lab"
readonly SERVICE_PATH="/etc/systemd/system"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER_NAME="$(hostname)-wsl2-rtx5060ti"

if [ "${EUID}" -ne 0 ]; then
  echo "Run this installer with sudo." >&2
  echo "  sudo bash scripts/github_actions/install_self_hosted_runner.sh" >&2
  exit 1
fi

for source_dir in "$REPOSITORY_ROOT/data" "$REPOSITORY_ROOT/ckpt"; do
  if [ ! -d "$source_dir" ]; then
    echo "Required local asset directory is missing: $source_dir" >&2
    exit 1
  fi
done

windows_mount_paths=()
for mount_path in /mnt/?; do
  if [ -d "$mount_path" ]; then
    windows_mount_paths+=("$mount_path")
  fi
done
if [ "${#windows_mount_paths[@]}" -eq 0 ]; then
  echo "No Windows drive mounts were found under /mnt; refusing to weaken runner isolation." >&2
  exit 1
fi
inaccessible_paths="${windows_mount_paths[*]}"
if [ -d /mnt/wslg ]; then
  inaccessible_paths+=" /mnt/wslg"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl ffmpeg git jq rclone tar util-linux

if ! getent group "$RUNNER_GROUP" >/dev/null; then
  groupadd --system "$RUNNER_GROUP"
fi
if ! id "$RUNNER_USER" >/dev/null 2>&1; then
  useradd \
    --system \
    --gid "$RUNNER_GROUP" \
    --create-home \
    --home-dir "$RUNNER_HOME" \
    --shell /bin/bash \
    "$RUNNER_USER"
fi

install -d -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0700 \
  "$STATE_ROOT" \
  "$RUNNER_HOME" \
  "$STATE_ROOT/assets" \
  "$STATE_ROOT/runs" \
  "$STATE_ROOT/training-queue"
for asset_dir in "$STATE_ROOT/assets/data" "$STATE_ROOT/assets/ckpt"; do
  if ! mountpoint --quiet "$asset_dir"; then
    install -d -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0700 "$asset_dir"
  fi
done
if [ ! -e "$STATE_ROOT/gpu.lock" ]; then
  install -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0600 /dev/null \
    "$STATE_ROOT/gpu.lock"
fi
install -d -o "$RUNNER_USER" -g "$RUNNER_GROUP" -m 0750 "$RUNNER_ROOT"
install -d -o root -g root -m 0755 "$TOOL_ROOT/bin"

TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

download_release_asset() {
  local repository="$1" asset_name="$2" destination="$3"
  local release_json asset_count url digest
  release_json="$(curl --fail --silent --show-error --location \
    "https://api.github.com/repos/$repository/releases/latest")"
  asset_count="$(jq --arg name "$asset_name" \
    '[.assets[] | select(.name == $name)] | length' <<< "$release_json")"
  if [ "$asset_count" -ne 1 ]; then
    echo "Expected one $asset_name asset in $repository latest release." >&2
    exit 1
  fi
  url="$(jq -r --arg name "$asset_name" \
    '.assets[] | select(.name == $name) | .browser_download_url' \
    <<< "$release_json")"
  digest="$(jq -r --arg name "$asset_name" \
    '.assets[] | select(.name == $name) | .digest' \
    <<< "$release_json")"
  if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "GitHub did not publish a usable SHA-256 digest for $asset_name." >&2
    exit 1
  fi
  curl --fail --silent --show-error --location "$url" --output "$destination"
  printf '%s  %s\n' "${digest#sha256:}" "$destination" | sha256sum --check --status
}

if [ ! -x "$RUNNER_ROOT/config.sh" ]; then
  runner_tag="$(curl --fail --silent --show-error --location \
    https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name)"
  runner_version="${runner_tag#v}"
  runner_asset="actions-runner-linux-x64-${runner_version}.tar.gz"
  download_release_asset "actions/runner" "$runner_asset" "$TEMP_ROOT/$runner_asset"
  tar -xzf "$TEMP_ROOT/$runner_asset" -C "$RUNNER_ROOT"
  chown -R "$RUNNER_USER:$RUNNER_GROUP" "$RUNNER_ROOT"
  "$RUNNER_ROOT/bin/installdependencies.sh"
fi

uv_tag="$(curl --fail --silent --show-error --location \
  https://api.github.com/repos/astral-sh/uv/releases/latest | jq -r .tag_name)"
uv_asset="uv-x86_64-unknown-linux-gnu.tar.gz"
download_release_asset "astral-sh/uv" "$uv_asset" "$TEMP_ROOT/$uv_asset"
mkdir "$TEMP_ROOT/uv"
tar -xzf "$TEMP_ROOT/$uv_asset" -C "$TEMP_ROOT/uv"
uv_binary="$(find "$TEMP_ROOT/uv" -type f -name uv -print -quit)"
uvx_binary="$(find "$TEMP_ROOT/uv" -type f -name uvx -print -quit)"
if [ -z "$uv_binary" ] || [ -z "$uvx_binary" ]; then
  echo "The uv release archive did not contain uv and uvx." >&2
  exit 1
fi
install -m 0755 "$uv_binary" /usr/local/bin/uv
install -m 0755 "$uvx_binary" /usr/local/bin/uvx
echo "Installed uv $uv_tag."

install -m 0755 \
  "$REPOSITORY_ROOT/.agents/skills/training-queue/scripts/training_queue.sh" \
  "$TOOL_ROOT/bin/training_queue.sh"
install -m 0755 \
  "$REPOSITORY_ROOT/.agents/skills/training-queue/scripts/prune_ckpts.py" \
  "$TOOL_ROOT/bin/prune_ckpts.py"
install -m 0755 "$SCRIPT_DIR/wsl_keepalive.sh" "$TOOL_ROOT/bin/wsl_keepalive.sh"

source_data="$(printf '%q' "$REPOSITORY_ROOT/data")"
source_ckpt="$(printf '%q' "$REPOSITORY_ROOT/ckpt")"
install -m 0755 /dev/stdin "$TOOL_ROOT/bin/mount_assets.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SOURCE_DATA=$source_data
SOURCE_CKPT=$source_ckpt
TARGET_ROOT="$STATE_ROOT/assets"

mount_readonly() {
  local source_path="\$1" target_path="\$2"
  if mountpoint --quiet "\$target_path"; then
    local mounted_root
    mounted_root="\$(findmnt --noheadings --output FSROOT --target "\$target_path" | xargs)"
    if [ "\$mounted_root" != "\$source_path" ]; then
      echo "Unexpected source for asset mount \$target_path: \$mounted_root" >&2
      exit 1
    fi
    if ! findmnt --noheadings --output OPTIONS --target "\$target_path" | grep -qw ro; then
      echo "Existing asset mount is not read-only: \$target_path" >&2
      exit 1
    fi
    return
  fi
  mount --bind "\$source_path" "\$target_path"
  mount --options remount,bind,ro "\$target_path"
}

case "\${1:-}" in
  start)
    mount_readonly "\$SOURCE_DATA" "\$TARGET_ROOT/data"
    mount_readonly "\$SOURCE_CKPT" "\$TARGET_ROOT/ckpt"
    ;;
  stop)
    if mountpoint --quiet "\$TARGET_ROOT/ckpt"; then
      umount "\$TARGET_ROOT/ckpt"
    fi
    if mountpoint --quiet "\$TARGET_ROOT/data"; then
      umount "\$TARGET_ROOT/data"
    fi
    ;;
  *)
    echo "usage: \$0 {start|stop}" >&2
    exit 2
    ;;
esac
EOF

install -m 0644 /dev/stdin "$SERVICE_PATH/tennis-lab-actions-assets.service" <<EOF
[Unit]
Description=Read-only tennis-lab data and checkpoint mounts for GitHub Actions
Before=tennis-lab-training-queue.service

[Service]
Type=oneshot
ExecStart=$TOOL_ROOT/bin/mount_assets.sh start
ExecStop=$TOOL_ROOT/bin/mount_assets.sh stop
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

install -m 0644 /dev/stdin "$SERVICE_PATH/tennis-lab-training-queue.service" <<EOF
[Unit]
Description=tennis-lab serial GPU training queue
Requires=tennis-lab-actions-assets.service
After=network-online.target tennis-lab-actions-assets.service
Wants=network-online.target

[Service]
Type=simple
User=$RUNNER_USER
Group=$RUNNER_GROUP
WorkingDirectory=$STATE_ROOT
Environment=HOME=$RUNNER_HOME
Environment=PATH=/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib
Environment=TRAINING_QUEUE_DIR=$STATE_ROOT/training-queue
Environment=TRAINING_QUEUE_LOCK_FILE=$STATE_ROOT/gpu.lock
ExecStart=$TOOL_ROOT/bin/training_queue.sh serve --idle-timeout 2147483647
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectProc=invisible
ProtectSystem=strict
ReadOnlyPaths=$STATE_ROOT/assets
ReadWritePaths=$RUNNER_HOME $STATE_ROOT/runs $STATE_ROOT/training-queue $STATE_ROOT/gpu.lock
InaccessiblePaths=$inaccessible_paths
RestrictSUIDSGID=true

[Install]
WantedBy=multi-user.target
EOF

if [ ! -f "$RUNNER_ROOT/.runner" ]; then
  if [ ! -t 0 ]; then
    echo "Runner registration requires an interactive one-hour GitHub token." >&2
    exit 1
  fi
  read -r -s -p "Paste the GitHub runner registration token: " registration_token
  echo
  if [ -z "$registration_token" ]; then
    echo "The registration token cannot be empty." >&2
    exit 1
  fi
  (
    cd "$RUNNER_ROOT"
    runuser -u "$RUNNER_USER" -- env HOME="$RUNNER_HOME" \
      ./config.sh \
      --unattended \
      --url "$REPOSITORY_URL" \
      --token "$registration_token" \
      --name "$RUNNER_NAME" \
      --labels "$RUNNER_LABELS" \
      --work _work \
      --replace
  )
  unset registration_token
fi

if [ ! -f "$RUNNER_ROOT/.service" ]; then
  (cd "$RUNNER_ROOT" && ./svc.sh install "$RUNNER_USER")
fi
runner_service="$(cat "$RUNNER_ROOT/.service")"
if [[ ! "$runner_service" =~ ^actions\.runner\.[A-Za-z0-9_.@-]+\.service$ ]]; then
  echo "Unexpected runner service name: $runner_service" >&2
  exit 1
fi

install -d -m 0755 "$SERVICE_PATH/$runner_service.d"
install -m 0644 /dev/stdin "$SERVICE_PATH/$runner_service.d/hardening.conf" <<EOF
[Unit]
Requires=tennis-lab-actions-assets.service
After=tennis-lab-actions-assets.service

[Service]
Environment=HOME=$RUNNER_HOME
Environment=PATH=/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectProc=invisible
ProtectSystem=strict
ReadOnlyPaths=$STATE_ROOT/assets
ReadWritePaths=$RUNNER_ROOT $RUNNER_HOME $STATE_ROOT/runs $STATE_ROOT/training-queue $STATE_ROOT/gpu.lock
InaccessiblePaths=$inaccessible_paths
RestrictSUIDSGID=true
EOF

systemctl daemon-reload
systemctl enable --now tennis-lab-actions-assets.service
systemctl enable --now tennis-lab-training-queue.service
systemctl enable --now "$runner_service"

runuser -u "$RUNNER_USER" -- env \
  HOME="$RUNNER_HOME" \
  PATH="/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib" \
  nvidia-smi --list-gpus
runuser -u "$RUNNER_USER" -- test -r "$STATE_ROOT/assets/ckpt/dino/checkpoint0029_4scale_swin.pth"

echo
echo "Runner service: $runner_service"
echo "Queue service:  tennis-lab-training-queue.service"
echo "State root:     $STATE_ROOT"
echo "Register the Windows logon keepalive task next; see scripts/github_actions/README.md."
