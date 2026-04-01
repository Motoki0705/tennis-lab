# DINO Checkpoints

This directory contains the official DINO 5-scale checkpoint and derived
backbone-only exports used in this workspace.

## Files

- `checkpoint0011_5scale.pth`: Official full DINO checkpoint from IDEA-Research.
- `checkpoint0027_5scale_swin.pth`: Official full DINO 5-scale Swin-L checkpoint
  from IDEA-Research.
- `backbone_body_state.pth`: Trimmed backbone body state dict with keys like
  `conv1.weight` and `layer4.2.conv3.weight`.
- `dino_backbone_module_state.pth`: `DINOBackbone.state_dict()` export with keys
  like `body.conv1.weight`.
- `swin_backbone_state_checkpoint0027_5scale.pth`: Trimmed Swin backbone state
  dict with keys like `patch_embed.proj.weight` and
  `layers.3.blocks.1.mlp.fc2.weight`.
- `dino_swin_backbone_module_state_checkpoint0027_5scale.pth`:
  `DINOSwinBackbone.state_dict()` export with keys like
  `backbone.patch_embed.proj.weight`.

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

To download the Swin-L 5-scale checkpoint:

```bash
python /workspace/checkpoints/DINO/scripts/download_checkpoint0027_5scale_swin.py
```

Expected checksum:

```text
17ddce1592816a0c63a2edc94d4a0877ffeb086f397a6657e151c703a4c850b5
```

## Backbone Extraction

To extract the ResNet backbone body state dict from the full checkpoint:

```bash
python /workspace/checkpoints/DINO/scripts/load_dino_backbone.py \
  --save-backbone-state /workspace/checkpoints/DINO/backbone_body_state.pth \
  --save-full-backbone-module /workspace/checkpoints/DINO/dino_backbone_module_state.pth \
  --strict
```


## Backbone Output Structure

To inspect the forward output structure of the loaded backbone:

```bash
python /workspace/checkpoints/DINO/scripts/inspect_dino_backbone_output.py --strict
```

With the default `800x800` dummy input, the backbone returns an `OrderedDict`
of four feature maps:

```text
[0] shape=(1, 256, 200, 200)
[1] shape=(1, 512, 100, 100)
[2] shape=(1, 1024, 50, 50)
[3] shape=(1, 2048, 25, 25)
```

These correspond to the ResNet `layer1` through `layer4` outputs. The loaded
backbone ends at `layer4`. In DINO's 5-scale setup, the fifth feature level is
not emitted by this backbone wrapper and is instead created later by the full
DINO model from the deepest backbone feature.

## Swin Backbone Extraction

To extract the Swin-L backbone from the full Swin checkpoint:

```bash
python /workspace/checkpoints/DINO/scripts/load_dino_swin_backbone.py \
  --save-backbone-state /workspace/checkpoints/DINO/swin_backbone_state_checkpoint0027_5scale.pth \
  --save-full-backbone-module /workspace/checkpoints/DINO/dino_swin_backbone_module_state_checkpoint0027_5scale.pth \
  --strict
```

Observed checkpoint metadata from the executed run:

- top-level keys: `model`, `optimizer`, `lr_scheduler`, `epoch`, `args`
- checkpoint size: `2615534683` bytes (`2.44 GB`)
- model key count: `722`
- extracted backbone key count: `357`
- resolved backbone config: `swin_L_384_22k`, `return_interm_indices=[0,1,2,3]`,
  `use_checkpoint=True`
- load result: `missing_keys=[]`, `unexpected_keys=[]`

## Swin Backbone Output Structure

To inspect the forward output structure of the loaded Swin backbone:

```bash
python /workspace/checkpoints/DINO/scripts/inspect_dino_swin_backbone_output.py \
  --checkpoint /workspace/checkpoints/DINO/swin_backbone_state_checkpoint0027_5scale.pth \
  --use-checkpoint \
  --device cuda \
  --height 384 \
  --width 384 \
  --strict
```

With the executed `384x384` dummy input on CUDA, the backbone returns an
`OrderedDict` of four feature maps:

```text
[0] shape=(1, 192, 96, 96)
[1] shape=(1, 384, 48, 48)
[2] shape=(1, 768, 24, 24)
[3] shape=(1, 1536, 12, 12)
```

As with the ResNet case, the Swin backbone itself emits four stages. The
detector's fifth feature level is produced later by the full DINO model through
`input_proj`.

## Swin Analysis

Ball-point feature consistency for the Swin checkpoint can be generated with the
shared analyzer:

```bash
python /workspace/checkpoints/DINO/analyze/ball_feature_consistency.py \
  --backbone swin_L_384_22k \
  --checkpoint /workspace/checkpoints/DINO/checkpoint0027_5scale_swin.pth \
  --clip-dir /workspace/data/tennis/game1/Clip1 \
  --device cuda \
  --strict
```

Generated artifacts:

- `ball_features.pt`: sampled per-frame features and load metadata.
- `ball_feature_consistency_per_frame.csv`: per-frame norms and cosine values.
- `ball_feature_consistency_summary.csv`: aggregated consistency metrics by scale.

The default output directory for this run is:

```text
/workspace/checkpoints/DINO/analyze/output/Clip1_swin_L_384_22k
```

## Notes

- The download source is the public Google Drive file linked from the
  IDEA-Research DINO release assets.
- `checkpoint0011_5scale.pth` is the full detector checkpoint, not just the
  backbone.
- `checkpoint0027_5scale_swin.pth` is also a full detector checkpoint, not just
  the backbone.
