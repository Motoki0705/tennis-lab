# SOLIDER inference port

Source: [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/blob/8c08e1c3255e8e1e51e006bf189e52cc57b009ed/model/backbones/swin_transformer.py), commit `8c08e1c3255e8e1e51e006bf189e52cc57b009ed`.
The upstream MIT license is preserved in [LICENSE](LICENSE).

Only the Swin inference graph is used. Removed MMCV/unused imports, checkpoint conversion/loading, stage-freezing training overrides and unused small/tiny factories. Semantic conditioning now follows the input device/dtype instead of unconditionally calling CUDA. The typed wrapper owns strict checkpoint loading; training/data/loss packages are not installed.

The MSMT17 Swin-Base recipe uses 384×128 RGB, mean/std 0.5, semantic weight 0.2 and the feature before the BN neck, L2-normalized. [Official config](https://github.com/tinyvision/SOLIDER-REID/blob/8c08e1c3255e8e1e51e006bf189e52cc57b009ed/configs/msmt17/swin_base.yml). The checkpoint supplies all backbone tensors; classifier/BN-neck tensors are checked but unused by this inference feature. No pretraining weights or network downloads are needed.
