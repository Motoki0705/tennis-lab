#!/usr/bin/env bash

# Codex entrypoint for the worktree-create skill.
# Source this file so the caller's shell can enter the created worktree.

_worktree_create_usage() {
  cat <<'USAGE'
Usage:
  source .agents/skills/worktree-create/scripts/enter_codex_worktree.sh \
    --name <worktree-name> \
    --branch <branch-name> \
    [--base <base-ref>]

Positional form is also supported:
  source .agents/skills/worktree-create/scripts/enter_codex_worktree.sh \
    <worktree-name> <branch-name> [base-ref]

Notes:
  - This script must be sourced; executing it cannot change the caller's cwd.
  - <worktree-name> becomes .claude/worktrees/<worktree-name>.
  - <branch-name> must be chosen outside this script.
  - <base-ref> defaults to origin/main.
USAGE
}

_worktree_create_fail() {
  echo "[FAIL] $*" >&2
  return 1
}

_worktree_create_main() {
  local name=""
  local branch=""
  local base_ref="origin/main"
  local positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        _worktree_create_usage
        return 0
        ;;
      --name)
        [[ $# -ge 2 ]] || { _worktree_create_fail "--name requires a value"; return 1; }
        name="$2"
        shift 2
        ;;
      --branch)
        [[ $# -ge 2 ]] || { _worktree_create_fail "--branch requires a value"; return 1; }
        branch="$2"
        shift 2
        ;;
      --base)
        [[ $# -ge 2 ]] || { _worktree_create_fail "--base requires a value"; return 1; }
        base_ref="$2"
        shift 2
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          positional+=("$1")
          shift
        done
        ;;
      --*)
        _worktree_create_fail "Unknown option: $1"
        return 1
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#positional[@]} -gt 0 && -z "$name" ]]; then
    name="${positional[0]}"
  fi
  if [[ ${#positional[@]} -gt 1 && -z "$branch" ]]; then
    branch="${positional[1]}"
  fi
  if [[ ${#positional[@]} -gt 2 && "$base_ref" == "origin/main" ]]; then
    base_ref="${positional[2]}"
  fi
  if [[ ${#positional[@]} -gt 3 ]]; then
    _worktree_create_fail "Too many positional arguments"
    return 1
  fi

  [[ -n "$name" ]] || { _worktree_create_fail "Missing worktree name"; return 1; }
  [[ -n "$branch" ]] || { _worktree_create_fail "Missing branch name"; return 1; }

  if [[ ${#name} -gt 64 ]]; then
    _worktree_create_fail "Worktree name must be at most 64 characters"
    return 1
  fi
  if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    _worktree_create_fail "Worktree name may contain only letters, digits, '.', '_', and '-'"
    return 1
  fi

  local repo_root=""
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    _worktree_create_fail "Not inside a git repository"
    return 1
  }

  local worktree_dir="${repo_root}/.claude/worktrees/${name}"

  if [[ -e "$worktree_dir" ]]; then
    _worktree_create_fail "Worktree path already exists: $worktree_dir"
    return 1
  fi

  if ! git -C "$repo_root" rev-parse --verify --quiet "${base_ref}^{commit}" >/dev/null; then
    _worktree_create_fail "Base ref does not resolve to a commit: $base_ref"
    return 1
  fi

  if git -C "$repo_root" show-ref --verify --quiet "refs/heads/${branch}"; then
    _worktree_create_fail "Local branch already exists: $branch"
    return 1
  fi

  mkdir -p "${repo_root}/.claude/worktrees" || return 1

  git -C "$repo_root" worktree add -b "$branch" "$worktree_dir" "$base_ref" || return 1
  cd "$worktree_dir" || return 1

  echo "[OK] Entered worktree: $PWD"
  echo "[OK] Branch: $(git branch --show-current)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[FAIL] This script must be sourced so it can change the caller's cwd." >&2
  echo "Usage: source .agents/skills/worktree-create/scripts/enter_codex_worktree.sh --name <worktree-name> --branch <branch-name> [--base <base-ref>]" >&2
  exit 1
fi

_worktree_create_main "$@"
_worktree_create_status=$?

unset -f _worktree_create_usage _worktree_create_fail _worktree_create_main
return "$_worktree_create_status"
