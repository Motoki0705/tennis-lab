# Official MSDA boundary

`OfficialMSDeformAttn` wraps the official Deformable DETR Python module and
CUDA autograd function; provenance and import-only changes are in
[vendor/origin.json](vendor/origin.json) and [vendor/NOTICE.md](vendor/NOTICE.md).
The native kernel is the existing DINO extension. The run manifest must verify
its CUDA/C++ source against official Deformable DETR and record build patches.

Build/select the binary with the [installation instructions](../../../../../../docs/installation/README.md#dino-cuda-extension).
Construction and state loading do not require CUDA. Execution requires the
explicit `TENNIS_LAB_DINO_EXTENSION`, CUDA, head width 64, valid shapes/starts,
and the official batch/im2col divisibility. CPU/reference fallback is absent.

FP32/BF16 input is explicitly cast to contiguous FP32; official value projection,
reference/offset/weight sampling and output projection run with autocast disabled.
The output is FP32, and gradients flow through casts. This is not BF16-native
MSDA. FP16/double input is rejected at this training boundary; the unchanged
low-level function can be checked with official double-precision gradcheck.

The boundary disables compilation explicitly; callers must allow graph breaks
and record them. The PyTorch reference in vendor is for verification only.
