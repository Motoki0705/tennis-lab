#!/usr/bin/env bash
set -euo pipefail

cd /root/repos/tennis-lab

# 使い方:
#   ./worktree_symbolic.sh                 # ブランチ名を自動生成
#   ./worktree_symbolic.sh feature/foo     # ブランチ名を指定
BRANCH="${1:-feature/wt-$(date +%Y%m%d-%H%M%S)}"
WT_PARENT="/root/repos/wt"
ORIG="$(pwd -P)"
BASE_REF="main"  # 新規ブランチは必ずmainから作成

# 現在のブランチがmainでない場合、mainをチェックアウト
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "現在のブランチ ($CURRENT_BRANCH) からmainに切り替えます..."
  git checkout main
  git pull origin main 2>/dev/null || echo "Warning: Could not pull from origin/main"
fi

mkdir -p "$WT_PARENT"

# そのブランチが既にどこかの worktree に割り当てられているか調べる
existing_wt="$(
  git worktree list --porcelain \
  | awk -v b="refs/heads/$BRANCH" '
      $1=="worktree"{wt=$2}
      $1=="branch" && $2==b {print wt; exit}
    '
)"

if [[ -n "${existing_wt}" ]]; then
  # 既にworktreeがある → そこに移動して開く
  cd "$existing_wt"
else
  # worktree未作成 → 作る
  WT_DIR="$WT_PARENT/${BRANCH##*/}"

  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    # ブランチは存在するが worktree は無い → そのブランチをcheckoutするworktreeを作る
    git worktree add "$WT_DIR" "$BRANCH"
  else
    # ブランチが無い → 新規ブランチ作成 + worktree 作成
    git worktree add -b "$BRANCH" "$WT_DIR" "$BASE_REF"
  fi

  cd "$WT_DIR"
fi

# シンボリックリンク生成（大きいものだけ共有）
rm -rf data .venv third_party outputs
ln -s "$ORIG/data" data
ln -s "$ORIG/.venv" .venv
ln -s "$ORIG/third_party" third_party
ln -s "$ORIG/outputs" outputs

# ※ third_party を symlink にしたいなら（trackedなら差分が出ます）
# rm -rf third_party
# ln -s "$ORIG/third_party" third_party

# VSCodeで開く
command -v code >/dev/null && code . || \
echo "code コマンドが無いです（VSCodeで 'Shell Command: Install code command in PATH' を実行）"
