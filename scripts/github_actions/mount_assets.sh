#!/usr/bin/env bash
# Only source paths are frozen; mount device IDs are re-read on every start.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/asset_paths.conf"

mount_readonly() {
  local source_path="$1" target_path="$2"
  if mountpoint --quiet "$target_path"; then
    local source_mount target_mount
    source_mount="$(findmnt --json --output FSROOT,TARGET,MAJ:MIN --target "$source_path")"
    target_mount="$(findmnt --json --output FSROOT,MAJ:MIN,OPTIONS --target "$target_path")"
    python3 -c '
import json, sys
from pathlib import PurePosixPath
source, target = [json.loads(value)["filesystems"] for value in sys.argv[1:3]]
if len(source) != 1 or len(target) != 1:
    sys.exit("Ambiguous asset mount")
s, t = source[0], target[0]
root = PurePosixPath(s["fsroot"])
if not root.is_absolute() or ".." in root.parts or "//" in str(root) or " (deleted)" in str(root):
    sys.exit("Unverifiable asset source root")
expected = str(root / PurePosixPath(sys.argv[3]).relative_to(s["target"]))
if t["fsroot"] != expected or t["maj:min"] != s["maj:min"] or "ro" not in t["options"].split(","):
    sys.exit("Unexpected source/device/options for asset mount")
' "$source_mount" "$target_mount" "$source_path"
    return
  fi
  mount --bind "$source_path" "$target_path"
  mount --options remount,bind,ro "$target_path"
  mount_readonly "$source_path" "$target_path"
}

case "${1:-}" in
  start)
    if ! mountpoint --quiet "$MIRROR_MOUNT"; then
      echo "E: mirror mount is unavailable: $MIRROR_MOUNT" >&2
      exit 1
    fi
    mount_readonly "$SOURCE_DATA" "$TARGET_ROOT/data"
    mount_readonly "$SOURCE_CKPT" "$TARGET_ROOT/ckpt"
    ;;
  stop)
    for name in ckpt data; do
      if mountpoint --quiet "$TARGET_ROOT/$name"; then
        umount "$TARGET_ROOT/$name"
      fi
    done
    ;;
  *) echo "usage: $0 {start|stop}" >&2; exit 2 ;;
esac
