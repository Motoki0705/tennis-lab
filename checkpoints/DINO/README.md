# DINO Checkpoints

This directory contains the official DINO 5-scale checkpoint and derived
backbone-only exports used in this workspace.

## Files

- `checkpoint0011_5scale.pth`: Official full DINO checkpoint from IDEA-Research.
- `backbone_body_state.pth`: Trimmed backbone body state dict with keys like
  `conv1.weight` and `layer4.2.conv3.weight`.
- `dino_backbone_module_state.pth`: `DINOBackbone.state_dict()` export with keys
  like `body.conv1.weight`.

## Download

Use the helper script below to download the official checkpoint into this
directory and verify its SHA256 checksum.

```bash
python /workspace/checkpoints/DINO/scripts/download_checkpoint0011_5scale.py
```

Expected checksum:

```text
1ccc1b6b7139813e4d3bfbeecfcf88347ebc226829769a0bf16c4a114c275cc0
```

To download to a different path:

```bash
python /workspace/checkpoints/DINO/scripts/download_checkpoint0011_5scale.py \
  --output /tmp/checkpoint0011_5scale.pth
```

## Backbone Extraction

To extract the ResNet backbone body state dict from the full checkpoint:

```bash
python /workspace/checkpoints/DINO/scripts/load_dino_backbone.py \
  --save-backbone-state /workspace/checkpoints/DINO/backbone_body_state.pth \
  --save-full-backbone-module /workspace/checkpoints/DINO/dino_backbone_module_state.pth \
  --strict
```

## Notes

- The download source is the public Google Drive file linked from the
  IDEA-Research DINO release assets.
- `checkpoint0011_5scale.pth` is the full detector checkpoint, not just the
  backbone.
