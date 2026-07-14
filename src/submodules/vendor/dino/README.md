# Vendored IDEA-Research DINO

This directory contains the inference-relevant model code from
`IDEA-Research/DINO` at commit `d84a491d41898b3befd8294d1cf2614661fc0953e`.
It is licensed under Apache-2.0; see `LICENSE`. Typed application code belongs
in `src/submodules/models/dino`, not here.

Local changes are limited to package-relative imports and PyTorch 2.12 CUDA
API compatibility (`Tensor.type()` to `Tensor.scalar_type()` in dispatch).

The custom CUDA op must be installed once in the project environment:

```bash
uv pip install -v --no-build-isolation ./src/submodules/vendor/dino/models/dino/ops
```

The wrapper fails explicitly if the extension or checkpoint is missing.
