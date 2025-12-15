#!/usr/bin/env bash
set -euo pipefail

codex_sandbox_env_setup() {
  local repo_root
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  export LOCAL_CACHE_DIR="${CODEX_LOCAL_CACHE_DIR:-$repo_root/agents_workspace/tmp_cache}"
  mkdir -p "$LOCAL_CACHE_DIR"

  export XDG_CACHE_HOME="$LOCAL_CACHE_DIR/xdg_cache"
  mkdir -p "$XDG_CACHE_HOME"

  export UV_CACHE_DIR="$LOCAL_CACHE_DIR/uv_cache"
  mkdir -p "$UV_CACHE_DIR"

  export PRE_COMMIT_HOME="$LOCAL_CACHE_DIR/pre_commit_home"
  mkdir -p "$PRE_COMMIT_HOME"

  export HOME="$LOCAL_CACHE_DIR/fake_home"
  mkdir -p "$HOME/.codex"

  if [[ -f "/root/.codex/auth.json" ]]; then
    cp -f "/root/.codex/auth.json" "$HOME/.codex/"
  fi
}
