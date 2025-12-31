#!/usr/bin/env bash
set -euo pipefail

# worktree削除スクリプト
# 使い方:
#   ./worktree_remove.sh <worktree-path1> [worktree-path2] [...]
#   ./worktree_remove.sh /root/repos/wt/wt-20231231-123456
#   ./worktree_remove.sh /root/repos/wt/wt-20231231-123456 /root/repos/wt/feature-foo

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <worktree-path1> [worktree-path2] [...]" >&2
  echo "Example: $0 /root/repos/wt/wt-20231231-123456" >&2
  echo "         $0 /root/repos/wt/wt-20231231-123456 /root/repos/wt/feature-foo" >&2
  exit 1
fi

ORIG_REPO="/root/repos/tennis-lab"

# 各ワークツリーに対して処理を実行
for WT_DIR in "$@"; do
  echo ""
  echo "========================================"
  echo "Processing: $WT_DIR"
  echo "========================================"
  
  if [[ ! -d "$WT_DIR" ]]; then
    echo "Error: Directory '$WT_DIR' does not exist (skipping)" >&2
    continue
  fi

  # worktree かどうか確認
  if ! git -C "$WT_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: '$WT_DIR' is not a valid git worktree (skipping)" >&2
    continue
  fi

  echo "==> Removing symbolic links in '$WT_DIR'..."

  # worktree内に移動
  cd "$WT_DIR"

  # シンボリックリンクを削除（worktree_symbolic.sh で作成されたもの）
  for link in data .venv third_party outputs; do
    if [[ -L "$link" ]]; then
      echo "  - Removing symlink: $link"
      rm -f "$link"
    elif [[ -e "$link" ]]; then
      echo "  - Warning: '$link' exists but is not a symlink (skipping)"
    fi
  done

  # サブモジュール関連のクリーンアップ（念のため）
  if [[ -f .gitmodules ]] || [[ -d .git/modules ]]; then
    echo "==> Cleaning up submodule references..."
    # .git/modules 内のサブモジュールディレクトリを削除
    if [[ -d .git/modules ]]; then
      rm -rf .git/modules
    fi
    # サブモジュールディレクトリ自体を削除（シンボリックの場合もあるため念のため）
    if [[ -f .gitmodules ]]; then
      git config --file .gitmodules --get-regexp '^submodule\..*\.path$' | \
        awk '{print $2}' | \
        while read -r sm_path; do
          if [[ -e "$sm_path" ]] || [[ -L "$sm_path" ]]; then
            echo "  - Removing submodule path: $sm_path"
            rm -rf "$sm_path"
          fi
        done
    fi
  fi

  # 元のリポジトリに戻る
  cd "$ORIG_REPO"

  echo "==> Removing worktree: $WT_DIR"
  git worktree remove "$WT_DIR" --force

  echo "==> Done! Worktree '$WT_DIR' removed successfully."
done

echo ""
echo "========================================"
echo "All worktrees processed!"
echo "========================================"
