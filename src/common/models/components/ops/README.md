# components.ops

`src.common.models.components.ops` is a backend-agnostic operator layer for custom kernels.

Goals:
- isolate extension-backed ops from high-level model code
- support CUDA/CPU/PyTorch fallback dispatch
- provide a consistent API for future operator families

Current operator family:
- `deformable`: multi-scale deformable attention primitives and module wrappers
