#!/usr/bin/env bash
# Install the reviewed agents.conf payload; never run the upstream latest installer.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
config_file="$repo_root/.agents/ops/lib/agents.conf"
bin_dir="/var/lib/tennis-lab-actions/home/.local/bin"
runner_install=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin-dir|--config-file)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      if [[ "$1" == --bin-dir ]]; then
        bin_dir="$2"
        runner_install=false
      else
        config_file="$2"
      fi
      shift 2
      ;;
    *)
      echo "usage: $0 [--bin-dir DIR] [--config-file FILE]" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -sm)" != "Linux x86_64" ]]; then
  echo "The pinned agy payload requires Linux x64." >&2
  exit 1
fi
if $runner_install; then
  if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo for the runner, or use --bin-dir for a local fixture." >&2
    exit 1
  fi
  id tennis-actions >/dev/null
fi

# Read assignments as data, using the same parser as agent_settings.py. Never source
# agents.conf or accept environment variables as overrides for these pins.
pins_text="$(python3 - "$repo_root/.agents/ops/lib" "$config_file" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from agy_headless import load_config

config = load_config(Path(sys.argv[2]))
print(config.version)
print(config.url)
print(config.sha256)
PY
)"
mapfile -t pins <<< "$pins_text"
agy_version="${pins[0]}"
agy_url="${pins[1]}"
agy_sha256="${pins[2]}"
target="$bin_dir/agy"
receipt="$bin_dir/.agy-install"

login_instructions() {
  cat <<'TEXT'
On the desktop, sign in interactively as the runner user (do not copy tokens):
  sudo -u tennis-actions env HOME=/var/lib/tennis-lab-actions/home AGY_CLI_DISABLE_AUTO_UPDATE=true /var/lib/tennis-lab-actions/home/.local/bin/agy
Complete the Google account sign-in in that terminal. Then run Actions → AGY smoke
on main as the repository owner. See scripts/github_actions/README.md.
TEXT
}

# A receipt is valid only while the installed bytes still match. Self-updates or
# manual replacements therefore cannot be mistaken for the pinned installation.
if [[ -f "$target" && ! -L "$target" && -x "$target" && -f "$receipt" ]]; then
  expected_receipt="$(printf '%s\n%s\n%s' "$agy_version" "$agy_sha256" "$(sha256sum "$target" | cut -d' ' -f1)")"
  if [[ "$(cat "$receipt")" == "$expected_receipt" ]]; then
    echo "agy $agy_version is already installed and unchanged."
    login_instructions
    exit 0
  fi
fi

temp_root="$(mktemp -d)"
staged_target=""
staged_receipt=""
trap 'rm -rf -- "$temp_root"; rm -f -- "$staged_target" "$staged_receipt"' EXIT
curl --fail --silent --show-error --location "$agy_url" --output "$temp_root/agy.tar.gz"
if ! printf '%s  %s\n' "$agy_sha256" "$temp_root/agy.tar.gz" | sha256sum --check --status; then
  echo "agy sha256 mismatch; installation aborted." >&2
  exit 1
fi
if [[ "$(tar -tzf "$temp_root/agy.tar.gz")" != antigravity ]]; then
  echo "agy archive must contain only antigravity." >&2
  exit 1
fi
tar -xOzf "$temp_root/agy.tar.gz" antigravity > "$temp_root/agy"
if [[ ! -s "$temp_root/agy" ]]; then
  echo "agy archive contains no executable bytes." >&2
  exit 1
fi

# Do not create or replace anything at the destination until verification passes.
if $runner_install; then
  install -d -o tennis-actions -g tennis-actions -m 0755 "$bin_dir"
else
  install -d -m 0755 "$bin_dir"
fi
staged_target="$(mktemp "$bin_dir/.agy.XXXXXX")"
staged_receipt="$(mktemp "$bin_dir/.agy-install.XXXXXX")"
install -m 0755 "$temp_root/agy" "$staged_target"
printf '%s\n%s\n%s\n' "$agy_version" "$agy_sha256" \
  "$(sha256sum "$temp_root/agy" | cut -d' ' -f1)" > "$staged_receipt"
chmod 0644 "$staged_receipt"
if $runner_install; then
  chown tennis-actions:tennis-actions "$staged_target" "$staged_receipt"
fi
mv -fT -- "$staged_target" "$target"
mv -fT -- "$staged_receipt" "$receipt"
echo "Installed pinned agy $agy_version."
login_instructions
