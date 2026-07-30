# PLCS synthetic-data libraries

This directory contains reusable Gaussian-avatar assets, controls, planning,
rendering, and visualization code for PLCS datasets. It is a domain library,
not a separate release pipeline.

Configure PLCS asset and renderer paths through the generic path manifest
described in
[`src/synthetic_data_generation/README.md`](../../README.md). Prepared inputs
belong under `third_party/nht/data`, intermediate artifacts under
`third_party/nht/artifacts`, final training data under `data`, and execution
logs/visualizations under `outputs`.

`components/avatar_control.py` provides the named GaussianAvatar-style and
HUGS-style control implementations. `rendering/` contains reusable NHT helpers,
and `visualization/` contains path-driven diagnostic functions. Quality metrics
are recorded by the generic pipeline and never determine whether later stages
run.
