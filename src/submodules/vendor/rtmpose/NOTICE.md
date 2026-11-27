# MMPose RTMPose-X inference layers

Copyright (c) OpenMMLab. All rights reserved.

Adapted from [MMPose at 7bea17cdf362cdd731fc5e79b982e4b79be5cca9](https://github.com/open-mmlab/mmpose/tree/7bea17cdf362cdd731fc5e79b982e4b79be5cca9).
See [LICENSE](LICENSE) and [origin.json](origin.json).

The adaptation provides the published Halpe26 RTMPose-X tensor forward with
plain PyTorch layers. It specializes CSPNeXt, spatial pyramid pooling, the
gated attention unit and the SimCC head to the checkpoint's settings. Standard
batch normalization in eval mode supplies the checkpoint's SyncBN computation.
Registry, training and data-pipeline code are outside this adaptation.
