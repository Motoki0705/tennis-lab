# KPR inference port

Source: [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification/tree/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa), commit `e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa`.
Authors: Vladimir Somers, Alexandre Alahi, Christophe De Vleeschouwer.
License: **Hippocratic License 3.0 HL3-LAW-MEDIA-MIL-SOC-SV**, full upstream text in LICENSE.
This portion is not relicensed under the repository MIT license.

`heads.py` contains only the inference pooling/dimension-reduction/classifier classes from
`torchreid/models/kpr.py`. Training, datasets, optimizers, Torchreid installation and pretrained
backbone loading are excluded. The typed wrapper fixes the public Market-1501/SOLIDER recipe:
384x128, mean/std .5, Swin-Base, six-part COCO prompts, five output parts + BN foreground,
MSF on, semantic weight -1, camera embeddings off. Shared Swin arithmetic is the separately
attributed MIT SOLIDER port. The duplicated image patch-embed state keys are preserved.
After-pooling dimension reduction flattens/restores leading dimensions arithmetically,
replacing upstream's forward shape branch to meet the repo's computation-only boundary.

Prompt generation follows upstream KeypointsToMasks/cck6/AddBackgroundMask: gaussian radius
floor(width/11), std=(2*radius+1)/4, confidence >.3, max within each body group, negative channel,
background clamp(1-sum,0,1). Out-of-crop joints are explicitly made invisible as in upstream's
inference transforms. Positive and optional negative prompts are in crop-normalized coordinates.

The checkpoint contains obsolete yacs config metadata. Loading is restricted to the recorded
SHA-256 of the official checkpoint; a local pickle class shim reads CfgNode as data without
installing yacs/Torchreid or modifying PyTorch. All 462 state keys, including unused classifiers,
are required with strict shape/key matching. No auxiliary backbone weight is loaded.
Native per-part embeddings and visibility are retained; they are not silently flattened into
a whole-image cosine encoder. Tracking/cross-camera integration must use an explicit part-distance
contract in a later comparison.
