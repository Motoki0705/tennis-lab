# Render-Supervised Global Court Geometry

`paper/src/court_detection` contains the self-contained English manuscript and
its Japanese section translation for the 3DGS-based Court Detection system.
The paper is structured after the
argument-first organization of `paper/reference/arXiv-2508.10104v1`, while its
text, figures, notation, and bibliography are independently authored.

The central claim is deliberately narrower than "zero annotation": after a
scene-level, fail-closed metric alignment has been accepted, NHT/3D Gaussian
Splatting renders supply RGB, intrinsics, camera-to-court 6DoF, geometric
validity, renderer support, and CourtKP14 supervision without per-frame manual
annotation.  The manuscript
describes the global court-query model from Issue #779 together with the
differentiable KP--pose consistency method specified in Issue #790 as one
completed manuscript method, as requested.  Repository evidence remains
separate: the #779 implementation is pinned in the paper, while no frozen #790
implementation or benchmark revision is claimed.  The manuscript does not
invent outcomes for experiments that have not been registered in `knowledge/`.

## Build

```bash
cd paper/src/court_detection
make
```

`make` builds both `main.pdf` (English, pdfLaTeX) and `main_jp.pdf` (Japanese,
LuaLaTeX/LuaTeX-ja).  Build either edition independently with `make main.pdf`
or `make main_jp.pdf`.  Intermediate products remain under `build/`;
`make clean` removes intermediates and `make distclean` also removes both PDFs.

## Source layout

| Path | Purpose |
|---|---|
| `main.tex` | English title and section order |
| `main_jp.tex` | Japanese title and section order; shares figures, tables, references, and appendices |
| `preamble.tex` | Packages, notation, colors, and shared macros |
| `sections/eg/` | English main manuscript, one argument per file; used by `main.tex` |
| `sections/jp/` | Japanese translation with matching filenames and LaTeX contracts |
| `appendices/` | Contracts, implementation details, and evaluation protocol |
| `figures/` | Repository-native TikZ figures |
| `tables/` | Reusable LaTeX tables |
| `refs.bib` | Verified BibTeX records cited by the manuscript |

## Evidence policy

- Implemented repository contracts are written in the present tense.
- A method may be described in the present tense while its code-evidence status
  is still printed as `specified`; method narrative and artifact provenance are
  intentionally distinct.
- Proposed numerical comparisons are presented as a preregistered protocol,
  not as completed measurements.
- Historical measurements are included only when their validation caveat is
  printed next to the value.
- `6DoF` refers to the camera-to-canonical-court rigid transform.  It does not
  refer to the yaw-only PLCS object label.
