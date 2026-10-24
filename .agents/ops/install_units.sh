#!/usr/bin/env bash
# Link the repo's systemd --user units (.agents/ops/*/systemd/*.{service,timer}) into
# ~/.config/systemd/user and reload. Timers and the dispatcher supervisor are
# only enabled with --enable.
#
# Usage: install_units.sh [--enable] [--dry-run]
#
# Units must reference the main checkout via %h/projects/tennis-lab (or TENNIS_LAB_ROOT
# set in the unit) so they keep working after worktrees are removed.
# The installer itself must also belong to the main checkout; linked sources are rejected.
set -euo pipefail

main() {
  local enable=0 dry_run=0
  while (($#)); do
    case "$1" in
      --enable) enable=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      *) echo "install_units: unknown argument: $1" >&2; exit 2 ;;
    esac
  done

  local ops_dir unit_dir git_dir common_dir
  ops_dir="$(cd "$(dirname "$0")" && pwd)"
  git_dir="$(git -C "$ops_dir" rev-parse --path-format=absolute --git-dir)"
  common_dir="$(git -C "$ops_dir" rev-parse --path-format=absolute --git-common-dir)"
  if [[ "$git_dir" != "$common_dir" ]]; then
    echo "install_units: refusing linked worktree source; use the installer in the main checkout" >&2
    exit 2
  fi
  unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

  # Retired units must be removed explicitly on the host, including dangling links
  # and wants/ aliases. Refuse before installing or reloading anything.
  # Keep the destination and source checks on the same retirement list.
  local -A retired_units=(
    [tennis-lab-cross-review.service]=.agents/ops/review/README.md
    [tennis-lab-cross-review.timer]=.agents/ops/review/README.md
    [tennis-lab-weekly-report.service]=.agents/ops/ops-report/README.md
    [tennis-lab-weekly-report.timer]=.agents/ops/ops-report/README.md
  )
  local retired found f
  for retired in "${!retired_units[@]}"; do
    if [[ -d "$unit_dir" ]]; then
      found="$(find "$unit_dir" -name "$retired" -print -quit)"
      if [[ -n "$found" ]]; then
        echo "install_units: retired unit remains: $found" >&2
        echo "Stop/disable/unlink it and daemon-reload as documented in ${retired_units[$retired]}, then rerun." >&2
        exit 2
      fi
    fi
  done

  # Enumerate into a file first so a find/sort failure aborts here (pipefail) instead of
  # being lost inside a process substitution and installing a partial set.
  UNIT_LIST="$(mktemp)"
  trap 'rm -f "$UNIT_LIST"' EXIT
  find "$ops_dir" -path '*/systemd/*' \( -name '*.service' -o -name '*.timer' \) -print0 | sort -z >"$UNIT_LIST"
  local -a units=()
  mapfile -d '' -t units <"$UNIT_LIST"
  ((${#units[@]})) || { echo "install_units: no units found under $ops_dir" >&2; exit 1; }

  for f in "${units[@]}"; do
    retired="$(basename "$f")"
    if [[ -n "${retired_units[$retired]:-}" ]]; then
      echo "install_units: refusing retired source unit: $f" >&2
      echo "Remove the retired source; see ${retired_units[$retired]}." >&2
      exit 2
    fi
  done

  ((dry_run)) || mkdir -p "$unit_dir"
  for f in "${units[@]}"; do
    echo "link $(basename "$f") -> $f"
    ((dry_run)) || ln -sfn "$f" "$unit_dir/$(basename "$f")"
  done
  ((dry_run)) && return 0

  systemctl --user daemon-reload
  if ((enable)); then
    for f in "${units[@]}"; do
      if [[ "$f" == *.timer || "$(basename "$f")" == tennis-lab-dispatch-supervisor.service ]]; then
        systemctl --user enable --now "$(basename "$f")"
      fi
    done
  fi
}

main "$@"
