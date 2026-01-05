#!/usr/bin/env bash
set -euo pipefail

# worktree削除スクリプト
# 使い方:
#   ./worktree_remove.sh <worktree-path1> [worktree-path2] [...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <worktree-path1> [worktree-path2] [...]" >&2
  echo "Example: $0 /home/motoki/repos/wt/feature-foo" >&2
  exit 1
fi

die() { echo "Error: $*" >&2; exit 1; }

# 対象 worktree が属する common .git / main worktree / admin gitdir を解決
resolve_ctx_from_wt() {
  local wt_dir="$1"
  local common_git_dir main_wt admin_gitdir

  common_git_dir="$(git -C "$wt_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
    || die "Failed to resolve git-common-dir for: $wt_dir"

  main_wt="$(
    git --git-dir "$common_git_dir" worktree list --porcelain \
      | awk '/^worktree /{print $2; exit}'
  )"
  [[ -n "$main_wt" ]] || die "Failed to resolve main worktree for: $wt_dir"

  # worktree list --porcelain から、その worktree の admin gitdir を引く（.git/worktrees/<id>）
  admin_gitdir="$(
    git --git-dir "$common_git_dir" worktree list --porcelain \
      | awk -v target="$wt_dir" '
          $1=="worktree"{wt=$2}
          $1=="gitdir"{gd=$2; if (wt==target){print gd; exit}}
        '
  )"

  echo "$common_git_dir|$main_wt|$admin_gitdir"
}

remove_symlinks() {
  local wt_dir="$1"
  echo "==> Removing symbolic links in '$wt_dir'..."

  for link in data .venv third_party outputs; do
    local p="$wt_dir/$link"
    if [[ -L "$p" ]]; then
      echo "  - Removing symlink: $link"
      rm -f "$p"
    elif [[ -e "$p" ]]; then
      echo "  - Warning: '$link' exists but is not a symlink (skipping)"
    fi
  done
}

cleanup_submodules_best_effort() {
  local wt_dir="$1"

  if [[ -f "$wt_dir/.gitmodules" ]]; then
    echo "==> Deinitializing submodules (best-effort)..."
    git -C "$wt_dir" submodule deinit -f --all >/dev/null 2>&1 || true

    # .gitmodules に書かれたパスの作業ツリー側を消す（worktree remove が submodule で落ちる回避）
    git -C "$wt_dir" config --file .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null \
      | awk '{print $2}' \
      | while read -r sm_path; do
          [[ -z "$sm_path" ]] && continue
          if [[ -e "$wt_dir/$sm_path" || -L "$wt_dir/$sm_path" ]]; then
            echo "  - Removing submodule path: $sm_path"
            rm -rf "$wt_dir/$sm_path"
          fi
        done || true
  fi
}

try_git_worktree_remove() {
  local common_git_dir="$1"
  local main_wt="$2"
  local wt_dir="$3"

  git --git-dir "$common_git_dir" --work-tree "$main_wt" worktree remove "$wt_dir" --force
}

manual_fallback_cleanup() {
  local common_git_dir="$1"
  local main_wt="$2"
  local admin_gitdir="$3"
  local wt_dir="$4"

  echo "==> Fallback: manual cleanup for '$wt_dir'..."
  rm -rf "$wt_dir"

  # admin gitdir が残っていたら消す（lock等で prune できないケースの保険）
  if [[ -n "$admin_gitdir" && -e "$admin_gitdir" ]]; then
    echo "  - Removing admin gitdir: $admin_gitdir"
    rm -rf "$admin_gitdir"
  fi

  git --git-dir "$common_git_dir" --work-tree "$main_wt" worktree prune --verbose >/dev/null 2>&1 || true
}

# ---- main ------------------------------------------------------------------

for WT_DIR_IN in "$@"; do
  echo ""
  echo "========================================"
  echo "Processing: $WT_DIR_IN"
  echo "========================================"

  if [[ ! -d "$WT_DIR_IN" ]]; then
    echo "Error: Directory '$WT_DIR_IN' does not exist (skipping)" >&2
    continue
  fi

  # 実パス化
  WT_DIR="$(cd "$WT_DIR_IN" && pwd -P)"

  if ! git -C "$WT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: '$WT_DIR' is not a valid git worktree (skipping)" >&2
    continue
  fi

  remove_symlinks "$WT_DIR"
  cleanup_submodules_best_effort "$WT_DIR"

  ctx="$(resolve_ctx_from_wt "$WT_DIR")"
  COMMON_GIT_DIR="${ctx%%|*}"
  rest="${ctx#*|}"
  MAIN_WT="${rest%%|*}"
  ADMIN_GITDIR="${rest#*|}"

  echo "==> Removing worktree: $WT_DIR"
  if try_git_worktree_remove "$COMMON_GIT_DIR" "$MAIN_WT" "$WT_DIR"; then
    echo "==> Done! Worktree '$WT_DIR' removed successfully."
  else
    echo "Warning: 'git worktree remove' failed. Trying fallback cleanup..." >&2
    manual_fallback_cleanup "$COMMON_GIT_DIR" "$MAIN_WT" "$ADMIN_GITDIR" "$WT_DIR"
    echo "==> Done (fallback). Worktree '$WT_DIR' cleaned."
  fi
done

echo ""
echo "========================================"
echo "All worktrees processed!"
echo "========================================"
