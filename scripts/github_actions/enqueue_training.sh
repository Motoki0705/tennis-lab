#!/usr/bin/env bash
# Create an isolated persistent checkout and enqueue one Actions training run.

set -euo pipefail

readonly EXPECTED_REPOSITORY="Motoki0705/tennis-lab"
readonly STATE_ROOT="/var/lib/tennis-lab-actions"
readonly ASSET_ROOT="$STATE_ROOT/assets"
readonly INSTALLED_QUEUE="/opt/tennis-lab-actions/bin/training_queue.sh"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
: "${GITHUB_RUN_ATTEMPT:?GITHUB_RUN_ATTEMPT is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${TRAINING_COMMAND:?TRAINING_COMMAND is required}"
: "${TRAINING_NAME:?TRAINING_NAME is required}"

TRAINING_ISSUE="${TRAINING_ISSUE:-}"

if [ "$GITHUB_REPOSITORY" != "$EXPECTED_REPOSITORY" ]; then
  echo "Refusing repository $GITHUB_REPOSITORY; expected $EXPECTED_REPOSITORY." >&2
  exit 1
fi
if [[ ! "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GITHUB_SHA must be a full commit SHA." >&2
  exit 1
fi
if [[ ! "$GITHUB_RUN_ID" =~ ^[0-9]+$ ]] || [[ ! "$GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]; then
  echo "GitHub run identifiers must be numeric." >&2
  exit 1
fi
if [ -n "$TRAINING_ISSUE" ] && [[ ! "$TRAINING_ISSUE" =~ ^[0-9]+$ ]]; then
  echo "TRAINING_ISSUE must be empty or a GitHub issue number." >&2
  exit 1
fi
if [ ! -x "$INSTALLED_QUEUE" ]; then
  echo "Installed training queue is unavailable: $INSTALLED_QUEUE" >&2
  exit 1
fi
if ! systemctl is-active --quiet tennis-lab-training-queue.service; then
  echo "tennis-lab-training-queue.service is not active." >&2
  exit 1
fi

run_root="$STATE_ROOT/runs/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
run_repository="$run_root/repository"
if [ -e "$run_root" ]; then
  echo "Persistent run directory already exists: $run_root" >&2
  exit 1
fi
install -d -m 0700 "$run_root"

git clone --no-local --no-checkout "$GITHUB_WORKSPACE" "$run_repository"
git -C "$run_repository" remote set-url origin \
  "https://github.com/$GITHUB_REPOSITORY.git"
git -C "$run_repository" checkout --detach "$GITHUB_SHA"
git -C "$run_repository" submodule update --init --recursive --depth 1
actual_sha="$(git -C "$run_repository" rev-parse HEAD)"
if [ "$actual_sha" != "$GITHUB_SHA" ]; then
  echo "Persistent checkout resolved to $actual_sha, expected $GITHUB_SHA." >&2
  exit 1
fi

for asset_name in data ckpt; do
  asset_source="$ASSET_ROOT/$asset_name"
  asset_target="$run_repository/$asset_name"
  if [ ! -d "$asset_source" ]; then
    echo "Runner asset mount is unavailable: $asset_source" >&2
    exit 1
  fi
  if [ -e "$asset_target" ] || [ -L "$asset_target" ]; then
    echo "Checkout unexpectedly contains $asset_name; refusing to replace it." >&2
    exit 1
  fi
  ln -s "$asset_source" "$asset_target"
done

queue_args=(
  add "$TRAINING_COMMAND"
  --name "$TRAINING_NAME"
  --provider github-actions
  --session "$GITHUB_RUN_ID"
)
if [ -n "$TRAINING_ISSUE" ]; then
  queue_args+=(--issue "$TRAINING_ISSUE")
fi
queue_output="$(
  cd "$run_repository"
  TRAINING_QUEUE_DIR="$STATE_ROOT/training-queue" \
    "$INSTALLED_QUEUE" "${queue_args[@]}"
)"
job_file="${queue_output#queued: }"
if [ "$job_file" = "$queue_output" ] || [[ ! "$job_file" =~ \.job$ ]]; then
  echo "Unexpected queue response: $queue_output" >&2
  exit 1
fi

echo "$queue_output"
echo "persistent checkout: $run_repository"
TRAINING_QUEUE_DIR="$STATE_ROOT/training-queue" "$INSTALLED_QUEUE" status

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Training queued"
    echo
    echo "- Queue job: \`$job_file\`"
    echo "- Commit: \`$GITHUB_SHA\`"
    echo "- Persistent checkout: \`$run_repository\`"
    echo
    echo "The workflow reports queue admission only; training continues in the local systemd queue."
  } >> "$GITHUB_STEP_SUMMARY"
fi
