# Person feature split reference

`person-features-before-split.npz` was generated with the pre-split
`FeatureExtractor` and `PersonTrackingModule` at commit `f7b016700`.
It contains 64 frames, 122 original detection rows and deterministic CPU model
outputs. Rows are sparse, poses reverse the ordering of two geometrically
identical people after frame 10, frames 15/16 and the trailing frame are empty.
The fake pose preserves raw confidence 1.4; the unprompted CLIP fake returns a
unit vector in the adopted 1280-dimensional space. AFLink returns identity
links. Real StrongSORT, GSI and court selection run without substitution.

`person-tracking-before-split.npz` and `person-selection-before-split.json`
freeze every tracking evidence array, GSI coordinates/masks, selection/group
axes and selection diagnostics produced by those original components.
The reference has IDs 1/2, four interpolated positions and 118 selected real
observations. Tests compare exact values after reloading the new feature
artifacts. Real model/GPU acceptance is separate from this CPU contract.
