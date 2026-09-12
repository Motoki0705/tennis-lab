# `src/automation`

`src/automation` contains repository-level automation that is neither reusable
domain logic (`src/utils`) nor a model task (`src/tasks`).

- `chatgpt_mcp/`: externalized ChatGPT execution control plane for a read-write tennis-lab sandbox, exact revisions, CUDA, and the logical two-slot training queue.
- `ci/`: test-file sharding and timing reports used by the development CLI; usage is documented in [`.spin/README.md`](../../.spin/README.md).
