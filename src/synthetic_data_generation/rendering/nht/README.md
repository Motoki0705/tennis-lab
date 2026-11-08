# Public NHT render data

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`NHTRenderClient` consumes only `nht-render` public files. Request/result schemas
and the CUDA runtime belong to [NHT](../../../../third_party/nht/README.md#standard-renderer).
The semantics below describe the adopted B00–B03 exports with view encoding enabled.

`rgb.npy` is float32 decoded RGB in the training-image encoding. The current
renderer accumulates harmonic features and applies a nonlinear learned shader;
it does **not** promise linear-light, straight-alpha or premultiplied-alpha RGB.
`alpha.npy` is the separate accumulated Gaussian opacity, not a matte identifying
the nearest object or the net.

`depth.npy` is float32 **expected camera z-depth** in canonical NHT scene units:
`sum(w_i * z_i) / sum(w_i)`. It is neither the nearest surface nor the distance
along the pixel ray. An empty accumulation produces zero. `nht_depth_to_metric`
divides by the explicit `nht_scene_units_per_metre`; this changes units only and
preserves the depth statistic. Near/far values come from the exported runtime
configuration and are expressed in the same NHT units.

These three arrays do not specify how to insert an opaque foreground inside a
partially transparent background. In particular, comparing foreground depth with
expected depth or treating the complete background alpha as net opacity is not a
supported occlusion contract. The current public request has no foreground-depth
input and the result has no depth-conditioned RGB contribution or transmittance.

The proposed contract and measured limitations are research evidence in
[the compositing study](../../../../knowledge/runs/run-player-depth-compositing-20261108/README.md),
not an implemented extension of these public schemas.
The [named approximation for C5](../../../../knowledge/runs/run-player-depth-compositing-20261108/contract.md#c5-approximation)
is also a proposal; it does not change the current array semantics above.
