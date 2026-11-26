#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
latexmk -norc -lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir=build main.tex

awk '
    /Missing character:|Overfull \\[hv]box|Warning:.*(undefined|multiply[- ]defined)|Rerun to get cross-references right/ {
        print > "/dev/stderr"
        failed = 1
    }
    END { exit failed }
' build/main.log

LC_ALL=C pdfinfo build/main.pdf | awk '/^Pages:/'
