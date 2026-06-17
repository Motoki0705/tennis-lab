# PLCS rotation-improvement experiments

Branch: `feat/plcs-rotation-improvement`

## Problem
Baseline (`loss=canonical`, `model=multiview_axial_base`, 100 ep, logs/version_2):
- `angular_error_deg`: mean **61.6**, median **45.4**
- `angle_accuracy`: 15deg **16.2%**, 30deg **33.6%**
- `position_error_m`: 0.260 (already good)
- Top errors were exact ~180deg front/back flips (`pred_rotation = -gt_rotation`).

## Root-cause findings (in order of discovery)
1. `rotation_weight=0.02` starved the rotation head → rotation broadly undertrained.
2. The `1-cos` rotation loss has a **flat saddle at 180deg** (grad = sin θ → 0 at
   the antipode). Verified: cos-loss grad@180 = [0,0], angle-loss grad = [0,1.0].
   Flips were a sticky equilibrium. Fix: add a wrapped-angle smooth-L1 `angle`
   term whose gradient stays constant out to 180deg.
3. Position and rotation **compete for shared-trunk capacity** (scalar loss
   reweighting could only trade one for the other).
4. Rotation is the HARD task: it needs a DEEP trunk **co-trained with the
   position + canonical tasks**. The position task in particular teaches multiview
   triangulation / cross-view correspondence that rotation depends on. Removing
   that signal (separate trunk / detach) collapses rotation, even with the
   canonical aux present.
5. **Resolution:** give rotation its own trunk but feed it the position signal via
   an *auxiliary* position head, while a *separate* trunk produces the precise
   final position. This removes the competition without starving rotation.

## Results (test split, 100 ep each)
| exp | model | loss | rot mean/med (deg) | acc@30 | pos mean/med (m) |
|-----|-------|------|--------------------|--------|------------------|
| baseline | base (shared) | canonical | 61.6 / 45.4 | 33.6% | 0.260 / 0.213 |
| exp1 | base (shared) | canonical_rot | 20.4 / 16.4 | 78.9% | 1.10 / 0.99 |
| exp2 | base (shared) | canonical_rot_v2 (pos30) | 52.0 / 35.6 | 43.7% | 0.30 / 0.23 |
| exp3 | branched (8+2+2) | canonical_rot | 13.6 / 9.7 | 88.6% | 0.82 / 0.68 |
| exp4 | branched | canonical_rot_v3 (pos30) | 54.1 / 26.9 | 52.9% | 0.40 / 0.29 |
| exp5 | split (0+6+6) | canonical_rot | 71.0 / 56.1 | 33.2% | 0.32 / 0.27 |
| exp6 | branched+detach | canonical_rot_v4 | 73.6 / 62.6 | 26.8% | 2.77 / 2.26 |
| exp7 | split, canon→rot | canonical_rot | 67.2 / 60.2 | 31.5% | 0.32 / 0.22 |
| exp8 | branched3 (8+3+3) | canonical_rot_v5 (pos6) | 12.5 / 9.9 | 93.7% | 0.60 / 0.47 |
| exp9 | branched3 | canonical_rot_v6 (pos12) | 49.5 / 25.5 | 54.9% | 0.59 / 0.51 |
| **exp10** | **split_auxpos** | **canonical_rot** | **9.98 / 7.25** | **95.7%** | **0.238 / 0.199** |

## Winner: exp10 (logs/version_15)
Beats baseline on BOTH axes simultaneously — no tradeoff:
- rotation **61.6 → 9.98 deg** mean (6.2x), median **45.4 → 7.25**; acc@30
  **33.6% → 95.7%**, acc@15 **16.2% → 80.3%**; raw rotation loss **0.607 → 0.028**.
- position **0.260 → 0.238 m** mean (slightly better); acc@0.5m 90.3% → 93.8%.
- The ~180deg front/back flips are essentially gone (median 7.25 deg).

### How to reproduce the winner
```
python -m src.tasks.plcs.scripts.train \
  data=multiview_sequence model=multiview_axial_base_split_auxpos \
  loss=canonical_rot training.trainer.max_epochs=100 data.batch_size=6
```
Recipe = three independent changes:
1. **`angle` loss term** (wrapped-angle smooth-L1) added to the registry; gives a
   non-vanishing gradient at the 180deg flip. Enabled via `loss=canonical_rot`
   (`angle_weight: 1.0`, `rotation_weight: 0.5`).
2. **Separate task trunks** (`num_layers: 0`, `num_task_layers: 6`): an
   independent rotation trunk and pose (position) trunk.
3. **Cross-task auxiliary heads on the rotation trunk**
   (`canonical_on_rotation_branch`, `aux_position_on_rotation_branch`): the
   rotation trunk also predicts canonical pose and an auxiliary position, so it
   learns the 3D-geometry + triangulation features rotation needs, while the
   pose trunk delivers precise position.

## Notes
- `batch_size=6` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids the
  intermittent OOM seen at bs=8 with long (up to 256-frame) variable sequences on
  the 16 GB GPU.
- Evaluation: `analyze_loss_dominance` on `logs/version_N/checkpoints/last.ckpt`
  plus the end-of-run test metrics in each `experiments/logs/*.log`.
