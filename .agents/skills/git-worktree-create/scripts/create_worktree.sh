#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  create_worktree.sh --branch <branch> [--base <ref>] [--parent <dir>]
  create_worktree.sh --issue <number> --topic <topic> [--prefix feat] [--base <ref>] [--parent <dir>]

Creates or reuses a git worktree for tennis-lab issue work and links shared paths:
  data, .venv, outputs, third_party

Environment:
  BASE_REF=main        Default base ref when --base is omitted.
  WT_PARENT=<dir>      Parent directory for worktrees.
  WT_FORCE_LINKS=1     Replace non-empty untracked link paths.
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 1
}

info() {
  echo "$*"
}

slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

resolve_repo_context() {
  local start_dir common_git_dir main_worktree

  start_dir="$(pwd -P)"
  common_git_dir="$(git -C "$start_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
    || die "Run this script from inside a git worktree."

  main_worktree="$(
    git --git-dir "$common_git_dir" worktree list --porcelain \
      | awk '$1=="worktree"{print $2; exit}'
  )"
  [[ -n "$main_worktree" && -d "$main_worktree" ]] \
    || die "Failed to resolve the primary worktree."

  printf '%s\n%s\n' "$common_git_dir" "$main_worktree"
}

find_worktree_for_branch() {
  local common_git_dir="$1"
  local branch="$2"

  git --git-dir "$common_git_dir" worktree list --porcelain \
    | awk -v ref="refs/heads/$branch" '
        $1=="worktree"{worktree=$2}
        $1=="branch" && $2==ref {print worktree; exit}
      '
}

resolve_base_ref() {
  local common_git_dir="$1"
  local main_worktree="$2"
  local base_ref="$3"
  local resolved="$base_ref"

  if git --git-dir "$common_git_dir" --work-tree "$main_worktree" remote get-url origin >/dev/null 2>&1; then
    if [[ "$base_ref" != origin/* && "$base_ref" != refs/* ]]; then
      git --git-dir "$common_git_dir" --work-tree "$main_worktree" fetch -q origin "$base_ref" || true
    fi
  fi

  if [[ "$base_ref" != origin/* ]] \
    && git --git-dir "$common_git_dir" --work-tree "$main_worktree" show-ref --verify --quiet "refs/remotes/origin/$base_ref"; then
    resolved="origin/$base_ref"
  fi

  git --git-dir "$common_git_dir" --work-tree "$main_worktree" rev-parse --verify --quiet "$resolved^{commit}" >/dev/null \
    || die "Base ref does not resolve to a commit: $resolved"

  printf '%s\n' "$resolved"
}

append_common_excludes() {
  local common_git_dir="$1"
  local exclude_file="$common_git_dir/info/exclude"
  local pattern

  mkdir -p "$(dirname "$exclude_file")"
  touch "$exclude_file"

  for pattern in /data /.venv /outputs /third_party; do
    grep -qxF "$pattern" "$exclude_file" || printf '%s\n' "$pattern" >> "$exclude_file"
  done
}

replace_with_symlink() {
  local target_worktree="$1"
  local source_worktree="$2"
  local name="$3"
  local link_path="$target_worktree/$name"
  local target_path="$source_worktree/$name"

  [[ -e "$target_path" || -L "$target_path" ]] \
    || die "Shared source path does not exist: $target_path"

  if [[ -L "$link_path" ]]; then
    rm -f "$link_path"
  elif [[ -e "$link_path" ]]; then
    if git -C "$target_worktree" ls-files --error-unmatch -- "$name" >/dev/null 2>&1; then
      die "Refusing to replace tracked path: $name"
    fi

    if [[ "${WT_FORCE_LINKS:-0}" == "1" ]]; then
      rm -rf "$link_path"
    elif [[ -d "$link_path" && -z "$(find "$link_path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "$link_path"
    else
      die "Refusing to replace non-empty path without WT_FORCE_LINKS=1: $link_path"
    fi
  fi

  ln -s "$target_path" "$link_path"
}

link_third_party() {
  local target_worktree="$1"
  local source_worktree="$2"
  local tracked_paths

  [[ -e "$source_worktree/third_party" || -L "$source_worktree/third_party" ]] \
    || die "Shared source path does not exist: $source_worktree/third_party"

  tracked_paths="$(git -C "$target_worktree" ls-files -- third_party || true)"
  if [[ -n "$tracked_paths" ]]; then
    while IFS= read -r path; do
      [[ -n "$path" ]] || continue
      git -C "$target_worktree" update-index --skip-worktree -- "$path"
    done <<< "$tracked_paths"
  fi

  if [[ -L "$target_worktree/third_party" ]]; then
    rm -f "$target_worktree/third_party"
  else
    rm -rf "$target_worktree/third_party"
  fi

  ln -s "$source_worktree/third_party" "$target_worktree/third_party"
}

BRANCH=""
ISSUE=""
TOPIC=""
PREFIX="feat"
BASE_REF="${BASE_REF:-main}"
WT_PARENT="${WT_PARENT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch|-b)
      [[ $# -ge 2 ]] || die "--branch requires a value"
      BRANCH="$2"
      shift 2
      ;;
    --issue)
      [[ $# -ge 2 ]] || die "--issue requires a value"
      ISSUE="$2"
      shift 2
      ;;
    --topic)
      [[ $# -ge 2 ]] || die "--topic requires a value"
      TOPIC="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "--prefix requires a value"
      PREFIX="$2"
      shift 2
      ;;
    --base)
      [[ $# -ge 2 ]] || die "--base requires a value"
      BASE_REF="$2"
      shift 2
      ;;
    --parent)
      [[ $# -ge 2 ]] || die "--parent requires a value"
      WT_PARENT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      if [[ -z "$BRANCH" ]]; then
        BRANCH="$1"
        shift
      else
        die "Unexpected argument: $1"
      fi
      ;;
  esac
done

if [[ -z "$BRANCH" ]]; then
  [[ -n "$ISSUE" && -n "$TOPIC" ]] \
    || die "Provide --branch, or provide both --issue and --topic."
  [[ "$ISSUE" =~ ^[0-9]+$ ]] || die "--issue must be a number: $ISSUE"
  TOPIC="$(slugify "$TOPIC")"
  [[ -n "$TOPIC" ]] || die "--topic must contain at least one letter or digit"
  BRANCH="$PREFIX/issue-$ISSUE-$TOPIC"
fi

BRANCH="${BRANCH#refs/heads/}"
git check-ref-format --branch "$BRANCH" >/dev/null 2>&1 \
  || die "Invalid branch name: $BRANCH"

mapfile -t repo_context < <(resolve_repo_context)
COMMON_GIT_DIR="${repo_context[0]}"
MAIN_WORKTREE="${repo_context[1]}"

if [[ -z "$WT_PARENT" ]]; then
  WT_PARENT="$(dirname "$MAIN_WORKTREE")/$(basename "$MAIN_WORKTREE").worktrees"
fi

mkdir -p "$WT_PARENT"

BASE_REF_RESOLVED="$(resolve_base_ref "$COMMON_GIT_DIR" "$MAIN_WORKTREE" "$BASE_REF")"
existing_worktree="$(find_worktree_for_branch "$COMMON_GIT_DIR" "$BRANCH" || true)"

if [[ -n "$existing_worktree" ]]; then
  WT_DIR="$existing_worktree"
  info "Reusing existing worktree: $WT_DIR"
else
  WT_SLUG="${BRANCH//\//__}"
  WT_DIR="$WT_PARENT/$WT_SLUG"

  if [[ -e "$WT_DIR" ]]; then
    if git -C "$WT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      current_branch="$(git -C "$WT_DIR" branch --show-current)"
      [[ "$current_branch" == "$BRANCH" ]] \
        || die "Directory is a git worktree for '$current_branch', not '$BRANCH': $WT_DIR"
      info "Reusing existing worktree directory: $WT_DIR"
    else
      die "Directory already exists and is not a git worktree: $WT_DIR"
    fi
  elif git --git-dir "$COMMON_GIT_DIR" --work-tree "$MAIN_WORKTREE" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git --git-dir "$COMMON_GIT_DIR" --work-tree "$MAIN_WORKTREE" worktree add "$WT_DIR" "$BRANCH"
  else
    git --git-dir "$COMMON_GIT_DIR" --work-tree "$MAIN_WORKTREE" worktree add -b "$BRANCH" "$WT_DIR" "$BASE_REF_RESOLVED"
  fi
fi

append_common_excludes "$COMMON_GIT_DIR"
replace_with_symlink "$WT_DIR" "$MAIN_WORKTREE" data
replace_with_symlink "$WT_DIR" "$MAIN_WORKTREE" .venv
replace_with_symlink "$WT_DIR" "$MAIN_WORKTREE" outputs
link_third_party "$WT_DIR" "$MAIN_WORKTREE"

info "branch=$BRANCH"
info "worktree=$WT_DIR"
info "base=$BASE_REF_RESOLVED"
info "linked=data .venv outputs third_party"
