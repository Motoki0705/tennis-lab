# 3DGS-native synthetic data

The current architecture and invocation guide live with the implementation in
[`src/synthetic_data_generation/README.md`](../../src/synthetic_data_generation/README.md).

This documentation directory retains visual assets from earlier experiments,
but those historical phase/release results are not pipeline configuration and
are not consumed by current code. The active pipeline is path-driven, writes
intermediate artifacts under `third_party/nht/artifacts`, and treats quality
metrics as observations rather than acceptance gates.
