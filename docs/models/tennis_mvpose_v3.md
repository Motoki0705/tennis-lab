# Tennis-DETR v3: Track-aware Scene Transformer

## Overview

Tennis-DETR v3 extends the v2/v2.5 hierarchical encoder with a **track-aware temporal encoder** over per-player query tracks. The model maintains the same task and I/O interface as v2/v2.5, enabling direct comparison while improving temporal consistency across frames.

## Key Differences from v2/v2.5

- **Track-aware temporal encoder**: After the standard decoder, per-player query tracks are processed with a TransformerEncoder across the time dimension.
- **Same hierarchical encoder**: Reuses the intra → inter → temporal encoder structure over detection tokens.
- **Decomposed outputs**: Maintains canonical pose, root translation, root rotation, and global pose heads.
- **Compatible I/O**: Uses the same input tensors and output keys as v2/v2.5.

## Architecture

### Scene Encoder (same as v2/v2.5)

1. **Token fusion**
   - Player keypoints (`[B,T,V,M,J,2]`) + court keypoints (`[B,V,C,2]`) → tokens (`[B,T,V,M,D]`)

2. **Hierarchical encoder**
   - `intra_encoder`: `[B*T*V, M, D]` (within-frame, per-camera)
   - `inter_encoder`: `[B*T, V*M, D]` (cross-camera within frame)
   - Result: `memory` (`[B, T*V*M, D]`)

### Decoder + Track-aware Temporal Encoder

1. **Decoder**
   - Queries: `num_queries (Q) × T`
   - `query_embed` + `time_embed` → `queries: [B, Q*T, D]`
   - `decoder(queries, memory)` → `dec_out: [B,Q,T,D]`

2. **Track-aware temporal encoder (v3 addition)**
   - Reshape to tracks: `[B*Q, T, D]`
   - `track_encoder(tracks)` → temporal refinement per query
   - Reshape back: `[B, Q, T, D]`

### Output Heads (decomposed)

- `canonical_head`: `[B,Q,T,D] → [B,Q,T,J,3]`
- `root_head`: `[B,Q,T,D] → [B,Q,T,5]` → split into `root_trans` (`[:3]`) and `root_rot` (`[3:]` → normalized 2D rotation)
- `exist_head`: `[B,Q,T,D] → [B,Q,1]` (logit)

Global pose is reconstructed by rotating the canonical pose with the 2D rotation and adding root translation.

## Configuration

All hyperparameters are defined in `TennisDetrV3Config` (see `src/models/tennis_multi_cam_3d_pose/config_v3.py`). Key parameters:

- `D_model`: Transformer dimension (default: 256)
- `nheads`: Number of attention heads (default: 8)
- `decoder_layers`: Decoder layers (default: 6)
- `intra_layers`, `inter_layers`, `temporal_layers`: Hierarchical encoder layers (default: 3 each)
- `num_queries`: Maximum number of players to track (default: 50)
- `max_frames`: Maximum sequence length (default: 32)

## Usage

### Model creation via factory

```python
from src.models.tennis_multi_cam_3d_pose import create_tennis_model, TennisDetrV3Config

cfg = TennisDetrV3Config()
model = create_tennis_model("v3", cfg)
```

### Training

Use the v3 training script:

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v3.sh
```

Or with overrides:

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v3.sh \
  --set training.trainer.max_epochs=50 \
  --set model.num_queries=30
```

### Evaluation

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v3.sh \
  --splits test \
  --num-samples 4
```

## Implementation Details

- **File locations**:
  - Model: `src/models/tennis_multi_cam_3d_pose/model_v3.py`
  - Config: `src/models/tennis_multi_cam_3d_pose/config_v3.py`
  - LightningModule: `src/training/tennis_multi_cam_3d_pose/lightning_v3.py`
  - Factory integration: `src/models/tennis_multi_cam_3d_pose/factory.py`
  - ConfigLoader integration: `src/training/utils/config.py`

- **Compatibility**:
  - Same dataset (`TennisPoseDataModule`) as v2/v2.5
  - Same loss formulation for direct comparison
  - Same evaluation pipeline (`src/evaluate/tennis_multi_cam_3d_pose.py`)

- **Design rationale**:
  - Track-aware encoding improves temporal consistency without changing the core task
  - Hierarchical encoder preserves multi-camera reasoning
  - Decomposed outputs maintain interpretability and training stability

## Comparison Summary

| Feature | v1 | v2 | v2.5 | v3 |
|---------|----|----|------|----|
| Encoder | Single | Hierarchical (intra→inter→temporal) | Hierarchical + explicit camera/time embeddings | Hierarchical |
| Decoder | Standard | Standard | Standard | Standard |
| Temporal modeling | None | Implicit via hierarchical encoder | Implicit | Explicit track-temporal encoder |
| Outputs | Direct pose_3d | Decomposed (canonical + root) | Decomposed | Decomposed |
| Camera/time embeddings | Implicit | Implicit | Explicit in encoder tokens | Implicit |
| Track awareness | No | No | No | Yes (per-query tracks) |
