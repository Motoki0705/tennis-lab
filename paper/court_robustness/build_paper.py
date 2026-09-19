"""Build the paper twice and bind the resulting PDF to its exact source files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from common import PAPER_PAGES, ROOT, sha256, write_json


def source_digests(root: Path) -> dict[str, str]:
    files = [root / "report.tex", *sorted((root / "figures").glob("*.png"))]
    return {str(p.relative_to(root)): sha256(p) for p in files}


def main() -> None:
    before = source_digests(ROOT)
    for _ in range(2):
        subprocess.run(
            ["lualatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    log = (ROOT / "report.log").read_text()
    errors = (
        "Missing character",
        "Overfull",
        "undefined references",
        "undefined on input",
    )
    if any(term in log for term in errors):
        raise ValueError("LaTeX output contains a layout, glyph or reference error")
    if source_digests(ROOT) != before:
        raise ValueError("Paper sources changed during the PDF build")
    info = subprocess.check_output(["pdfinfo", str(ROOT / "report.pdf")], text=True)
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if match is None or int(match.group(1)) != PAPER_PAGES:
        raise ValueError(f"The paper must fit {PAPER_PAGES} pages without overflow")
    write_json(
        ROOT / "evidence/build.json",
        {
            "schema": "court_paper_build_v1",
            "source_sha256": before,
            "pdf_sha256": sha256(ROOT / "report.pdf"),
            "pages": PAPER_PAGES,
            "latex_runs": 2,
            "layout_glyph_reference_checks": "passed",
            "compiler": subprocess.check_output(
                ["lualatex", "--version"], text=True
            ).splitlines()[0],
        },
    )
    print(
        f"Built {PAPER_PAGES}-page PDF; source/figure/PDF hashes saved in evidence/build.json"
    )


if __name__ == "__main__":
    main()
