#!/usr/bin/env bash
set -euo pipefail

# 使い方:
#   ./worktree_symbolic.sh                 # ブランチ名を自動生成
#   ./worktree_symbolic.sh feature/foo     # ブランチ名を指定
#
# 環境変数:
#   WT_PARENT=/path/to/wt-parent           # worktree を作る親ディレクトリ（省略可）
#   BASE_REF=main                          # 新規ブランチ作成の基点（省略可）

# ---- helpers ---------------------------------------------------------------

die() { echo "Error: $*" >&2; exit 1; }

# スクリプトの場所から、その worktree が属する「共通 .git」と「main worktree」を解決する
resolve_repo_context() {
  local script_dir common_git_dir main_wt

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

  # common git dir（例: /path/to/main/.git）
  common_git_dir="$(git -C "$script_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
    || die "This script must be located inside a git working tree."

  # main worktree は worktree list の先頭に出る（通常）
  main_wt="$(
    git --git-dir "$common_git_dir" worktree list --porcelain \
      | awk '/^worktree /{print $2; exit}'
  )"
  [[ -n "$main_wt" && -d "$main_wt" ]] || die "Failed to resolve main worktree."

  echo "$common_git_dir|$main_wt"
}

# その branch がすでにどこかの worktree に割り当てられているなら、そのパスを返す
find_worktree_for_branch() {
  local common_git_dir="$1"
  local main_wt="$2"
  local branch="$3"

  git --git-dir "$common_git_dir" --work-tree "$main_wt" worktree list --porcelain \
    | awk -v b="refs/heads/$branch" '
        $1=="worktree"{wt=$2}
        $1=="branch" && $2==b {print wt; exit}
      '
}

safe_replace_with_symlink() {
  local link_path="$1"
  local target_path="$2"

  if [[ -L "$link_path" ]]; then
    rm -f "$link_path"
  elif [[ -e "$link_path" ]]; then
    # 既存がディレクトリ/ファイルの場合は削除（この挙動が嫌ならここを保守的に変更）
    rm -rf "$link_path"
  fi

  ln -s "$target_path" "$link_path"
}

# ---- main ------------------------------------------------------------------

BRANCH="${1:-feature/wt-$(date +%Y%m%d-%H%M%S)}"
BRANCH="${BRANCH#refs/heads/}"  # 念のため

# branch 名の妥当性チェック
git check-ref-format --branch "$BRANCH" >/dev/null 2>&1 \
  || die "Invalid branch name: $BRANCH"

BASE_REF="${BASE_REF:-main}"

ctx="$(resolve_repo_context)"
COMMON_GIT_DIR="${ctx%%|*}"
ORIG="${ctx#*|}"   # main worktree（データ共有の起点）

WT_PARENT_DEFAULT="$(dirname "$ORIG")/wt"
WT_PARENT="${WT_PARENT:-$WT_PARENT_DEFAULT}"

mkdir -p "$WT_PARENT"

# origin があれば、BASE_REF をなるべく最新化（checkout しない）
if git --git-dir "$COMMON_GIT_DIR" --work-tree "$ORIG" remote get-url origin >/dev/null 2>&1; then
  git --git-dir "$COMMON_GIT_DIR" --work-tree "$ORIG" fetch -q origin "$BASE_REF" || true
fi

# base ref は origin/<BASE_REF> があればそれを優先
BASE_REF_RESOLVED="$BASE_REF"
if git --git-dir "$COMMON_GIT_DIR" --work-tree "$ORIG" show-ref --verify --quiet "refs/remotes/origin/$BASE_REF"; then
  BASE_REF_RESOLVED="origin/$BASE_REF"
fi

# そのブランチがすでに worktree に存在するか
existing_wt="$(find_worktree_for_branch "$COMMON_GIT_DIR" "$ORIG" "$BRANCH" || true)"

if [[ -n "$existing_wt" ]]; then
  cd "$existing_wt"
else
  WT_SLUG="${BRANCH//\//-}"        # branch の / を - にして衝突しにくく
  WT_DIR="$WT_PARENT/$WT_SLUG"

  # すでにディレクトリがある場合は「それが worktree なら開く / 違うならエラー」
  if [[ -d "$WT_DIR" ]]; then
    if git -C "$WT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      cd "$WT_DIR"
    else
      die "Directory already exists but is not a git worktree: $WT_DIR"
    fi
  else
    # ブランチが既にあるか？
    if git --git-dir "$COMMON_GIT_DIR" --work-tree "$ORIG" show-ref --verify --quiet "refs/heads/$BRANCH"; then
      git --git-dir "$COMMON_GIT_DIR" --work-tree "$ORIG" worktree add "$WT_DIR" "$BRANCH"
    else
      git --git-dir "$COMMON_GIT_DIR" --work-tree "$ORIG" worktree add -b "$BRANCH" "$WT_DIR" "$BASE_REF_RESOLVED"
    fi
    cd "$WT_DIR"
  fi
fi

# シンボリックリンク生成（大きいものだけ共有）
# ※ third_party が tracked/submodule の場合、差分が出たり運用ポリシーが必要です
safe_replace_with_symlink "$PWD/data"       "$ORIG/data"
safe_replace_with_symlink "$PWD/.venv"      "$ORIG/.venv"
safe_replace_with_symlink "$PWD/third_party" "$ORIG/third_party"
safe_replace_with_symlink "$PWD/outputs"    "$ORIG/outputs"

# VSCodeで開く
if command -v code >/dev/null 2>&1; then
  code .
else
  echo "code コマンドが無いです（VSCodeで 'Shell Command: Install code command in PATH' を実行）" >&2
fi
