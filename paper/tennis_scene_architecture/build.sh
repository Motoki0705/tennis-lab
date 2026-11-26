#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )) || [[ $# == 1 && $1 != --check-tracked ]]; then
    printf 'Usage: %s [--check-tracked]\n' "$0" >&2
    exit 2
fi

check_tracked_inputs() {
    local root=$1
    local input path missing=0
    while IFS= read -r input; do
        path=$(realpath -ms -- "$input")
        [[ "$path" == "$root/"* ]] || continue
        path=${path#"$root/"}
        if ! git -C "$root" --literal-pathspecs ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
            printf 'Untracked LaTeX input: %s\n' "$path" >&2
            missing=1
        fi
    done
    return "$missing"
}

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dependency_args=()
if (( $# )); then
    repo_root=$(git rev-parse --show-toplevel)
    mkdir -p build
    dependency_args=(-deps-out=build/main.deps -deps-escape=none)
fi
latexmk -norc -lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir=build "${dependency_args[@]}" main.tex

awk '
    /Missing character:|Overfull \\[hv]box|Warning:.*(undefined|multiply[- ]defined)|Rerun to get cross-references right/ {
        print > "/dev/stderr"
        failed = 1
    }
    END { exit failed }
' build/main.log

if (( $# )); then
    # The latexmk dependency list has one indented input per line.
    awk '
        sub(/^    /, "") {
            sub(/\\$/, "")
            print
            found = 1
        }
        END {
            if (!found) {
                print "No inputs in build/main.deps" > "/dev/stderr"
                exit 1
            }
        }
    ' build/main.deps | check_tracked_inputs "$repo_root"
fi

LC_ALL=C pdfinfo build/main.pdf | awk '/^Pages:/'
