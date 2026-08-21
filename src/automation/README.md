# `src/automation`

`src/automation` contains repository-level automation that is neither reusable
domain logic (`src/utils`) nor a model task (`src/tasks`).

- `chatgpt_mcp/`: externalized ChatGPT execution control plane for a read-write tennis-lab sandbox, exact revisions, CUDA, and the serial training queue.
- [`codex_trace/`](codex_trace/README.md): local, inference-level token and tool-I/O analysis for opt-in Codex rollout-trace bundles.
