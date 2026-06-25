#!/usr/bin/env bash
# worktree-guard.sh — assert that a directory is a tennis-lab git *worktree*, not the
# main tree, before anything edits there.
#
# Why: worktrees in this repo are nested at <repo-root>/.claude/worktrees/<name>/, so
# the same relative path resolves to a different file depending on the working
# directory. An agent that thinks it is "in the worktree" but whose working dir is the
# repo root silently edits `main`. This guard is the cheap pre-flight that catches it.
#
# Usage:
#   worktree-guard.sh [DIR]        # DIR defaults to the current directory
#
# Exit codes:
#   0  DIR is a worktree under .claude/worktrees/ on a non-main branch -> safe to edit
#   1  DIR is the main tree, or on main/master                         -> STOP
#   2  DIR is not inside a git repository / bad usage
#
# Claude Code: run in Bash before your first edit after EnterWorktree.
# Codex CLI:   run as a pre-flight on the same DIR you pass to `codex exec -C DIR`
#              (i.e. `codex-auto.sh --dir DIR`).
set -euo pipefail

DIR="${1:-$PWD}"

if ! top="$(git -C "$DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "STOP: '$DIR' is not inside a git repository" >&2
  exit 2
fi
branch="$(git -C "$DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"

case "$top" in
  */.claude/worktrees/*)
    if [[ "$branch" == "main" || "$branch" == "master" ]]; then
      echo "STOP: worktree '$top' is on '$branch' — refusing to edit main/master" >&2
      exit 1
    fi
    echo "OK: worktree -> $top (branch: $branch)"
    ;;
  *)
    echo "STOP: '$top' is the MAIN tree (branch: $branch); do not edit here." >&2
    echo "      For worktree work, point your working dir / --dir at" >&2
    echo "      <repo-root>/.claude/worktrees/<name>/ instead." >&2
    exit 1
    ;;
esac
