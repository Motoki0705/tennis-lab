# Path-driven synthetic-data generation

This package builds synthetic datasets from configured filesystem paths. It is
not a release-acceptance or artifact-lineage system: downstream stages consume
the paths written to one shared manifest, and measured quality is descriptive
output rather than permission to continue.

## Directory ownership

```text
synthetic_data_generation/
├── alignment/       # Court/scene geometry and fitting libraries
├── composition/     # Gaussian composition libraries
├── dataset/         # Generic path manifest, execution, and domain algorithms
├── rendering/       # Renderer adapters and runtime path helpers
├── visualization/   # Generic path/metrics/render summaries
├── scripts/         # User-facing executable entry points
└── configs/         # Hydra configuration for those entry points
```

Domain algorithms under `dataset/{blcs,plcs,court}` remain reusable libraries.
The pipeline itself is generic and does not select a domain, experiment, phase,
or release cycle.

## Path convention

The default config provides an automatic flow with no manual artifact moving:

```text
third_party/nht/data/
    alignment-observations.json
    render-jobs.json
    prepared renderer inputs and references
             │
             ▼
third_party/nht/artifacts/synthetic-data/
    alignment-metrics.json
    dataset-plan.json
    render-manifest.json
    quality-metrics.json
             │
             ├──► data/synthetic_data_generation/       final dataset files
             └──► outputs/synthetic_data_generation/    run logs and HTML summary
```

Every path can be overridden in
`configs/dataset/pipeline.yaml` or with a Hydra override. Relative paths are
resolved once against `project_root`, then stored as absolute paths in
`path-manifest.json`.

## Inputs

`alignment-observations.json` contains numeric residuals:

```json
{
  "residuals": [0.02, -0.01, 0.03]
}
```

`render-jobs.json` contains named path mappings. `input` and `reference` are
relative to `source_root`; `output` is relative to `dataset_root` unless an
absolute path is configured.

```json
{
  "jobs": [
    {
      "name": "sample-0001",
      "input": "prepared/sample-0001.bin",
      "output": "renders/sample-0001.bin",
      "reference": "references/sample-0001.bin",
      "arguments": []
    }
  ]
}
```

Set `renderer.command` to a shell-free argv list. Tokens may use `{input}`,
`{output}`, `{reference}`, `{source_root}`, `{artifact_root}`, or
`{dataset_root}`. An empty command copies prepared render inputs to their
configured final paths, which is useful when rendering happened in an external
batch.

## Running

Write only the path manifest:

```bash
.venv/bin/python -m \
  src.synthetic_data_generation.scripts.dataset.run_pipeline
```

Run alignment metrics, planning, rendering, quality metrics, and visualization:

```bash
.venv/bin/python -m \
  src.synthetic_data_generation.scripts.dataset.run_pipeline execute=true
```

Metrics are always written, including poor values. A run stops only for a
missing configured input, malformed JSON, an invalid input shape, a renderer
failure, or a renderer that does not produce its configured output.

The generated HTML summary is generic: it reads the shared manifest and its
configured artifacts, with no hard-coded experiment or cycle paths.
