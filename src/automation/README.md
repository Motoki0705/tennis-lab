# `src/automation`

`src/automation` contains repository-level automation that is neither reusable
domain logic (`src/utils`) nor a model task (`src/tasks`).

- `chatgpt_mcp/`: authenticated ChatGPT MCP access to an isolated tennis-lab
  workspace, CUDA-capable command jobs, and the repository training queue.
