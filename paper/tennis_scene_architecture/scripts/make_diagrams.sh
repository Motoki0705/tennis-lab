#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p ../build/diagrams
# Keep the PDF dates fixed so regenerating a diagram does not change its bytes.
export SOURCE_DATE_EPOCH=0
export FORCE_SOURCE_DATE=1
for diagram in pipeline execution; do
    lualatex -interaction=nonstopmode -halt-on-error -file-line-error         -output-directory=../build/diagrams "$diagram.tex"
    install -m 644 "../build/diagrams/$diagram.pdf" "../figures/$diagram.pdf"
done
