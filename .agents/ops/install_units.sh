#!/usr/bin/env bash
# Link the repo's systemd --user units (.agents/ops/*/systemd/*.{service,timer}) into
# ~/.config/systemd/user and reload. Timers are only enabled with --enable.
#
# Usage: install_units.sh [--enable] [--dry-run]
#
# Units must reference the main checkout via %h/projects/tennis-lab (or TENNIS_LAB_ROOT
# set in the unit) so they keep working after worktrees are removed.
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

  local ops_dir unit_dir
  ops_dir="$(cd "$(dirname "$0")" && pwd)"
  unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

  local -a units=()
  local f
  while IFS= read -r -d '' f; do units+=("$f"); done < <(
    find "$ops_dir" -path '*/systemd/*' \( -name '*.service' -o -name '*.timer' \) -print0 | sort -z
  )
  ((${#units[@]})) || { echo "install_units: no units found under $ops_dir" >&2; exit 1; }

  for f in "${units[@]}"; do
    echo "link $(basename "$f") -> $f"
    ((dry_run)) || ln -sfn "$f" "$unit_dir/$(basename "$f")"
  done
  ((dry_run)) && return 0

  systemctl --user daemon-reload
  if ((enable)); then
    for f in "${units[@]}"; do
      [[ "$f" == *.timer ]] && systemctl --user enable --now "$(basename "$f")"
    done
  fi
}

main "$@"
