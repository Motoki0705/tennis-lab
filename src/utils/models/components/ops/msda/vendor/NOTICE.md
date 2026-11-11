Python module and autograd function from fundamentalvision/Deformable-DETR,
the version recorded in origin.json (Apache-2.0).

The MSDeformAttn module is unchanged. The autograd function only replaces the
native extension import with an explicit lazy `require_extension()` call in
forward/backward. Sampling, gradients and the test-only PyTorch reference are
unchanged. CUDA/C++ comes from the pinned DINO submodule; the experiment build
manifest compares its bytes with the official source and records ABI patches.
