# Tennis Pose Training パイプライン仕様（Spec）

本書は、テニス用マルチビュー 3D ポーズ推定の学習パイプライン（ConfigLoader, DataModule, LightningModule, CLI）の仕様をまとめる。

対象:
- `src/training/utils/config.py:ConfigLoader`（`task == "tennis_pose"` 分岐）
- `src/training/tennis/datamodule.py:TennisPoseDataModule`
- `src/training/tennis/lightning.py:TennisDetrModule`
- `src/cli/train_tennis_pose.py`
- YAML 設定: `configs/tennis_pose.yaml` およびその includes

---

## 1. ワークフロー（概要）

擬似コード:

```python
from src.training.utils.config import load_cfg, ConfigLoader

cfg     = load_cfg("configs/tennis_pose.yaml", overrides)
loader  = ConfigLoader(cfg)
dm      = loader.build_datamodule()       # TennisPoseDataModule
lit     = loader.build_lit_module()       # TennisDetrModule
logger  = loader.build_logger()           # TensorBoardLogger
cbs     = loader.build_callbacks()        # Checkpoint, LR monitor など
trainer = loader.build_trainer(logger, cbs)

trainer.fit(lit, datamodule=dm)
```

CLI では `src/cli/train_tennis_pose.py` がこのフローをラップする。

---

## 2. ConfigLoader の `tennis_pose` 分岐

実装: `src/training/utils/config.py:ConfigLoader`

### 2.1 `build_datamodule`

```python
def build_datamodule(self) -> DancetrackDataModule | TennisPoseDataModule:
    task = self._task()
    dataset_cfg = self.cfg.get("dataset")
    debug_cfg = self.cfg.get("debug")
    if task == "tennis_pose":
        from src.training.tennis.datamodule import TennisPoseDataModule
        return TennisPoseDataModule(dataset_cfg, debug_cfg)
    ...
```

### 2.2 `build_lit_module`

```python
def build_lit_module(self) -> SceneModelLightningModule | TennisDetrModule:
    task = self._task()
    if task == "tennis_pose":
        from src.training.tennis.lightning import TennisDetrModule
        return TennisDetrModule(self.cfg)
    ...
```

その他のメソッド（`build_logger`, `build_callbacks`, `build_trainer`）は SceneModel 用と共通で、`cfg.logging` / `cfg.training.trainer` を解釈して TensorBoardLogger / ModelCheckpoint / LRMonitor / Trainer を構築する。

---

## 3. YAML 設定構成

### 3.1 トップレベル: `configs/tennis_pose.yaml`

```yaml
task: tennis_pose
experiment_name: tennis_mvpose_dev

includes:
  dataset: configs/datasets/tennis_pose_sim.yaml
  model: configs/models/tennis_mvpose.yaml
  training: configs/training/tennis_mvpose.yaml
  logging: configs/logging/tennis_mvpose.yaml
```

- `load_cfg` が `includes` を展開し、`cfg.dataset`, `cfg.model`, `cfg.training`, `cfg.logging` を構成する。

### 3.2 データセット: `configs/datasets/tennis_pose_sim.yaml`

代表例:

```yaml
root: data/tennis_autogen
name: sim_fps60_dur3p0_C4_P1-20_T10
window_T: 10
max_cameras: 4
max_players: 20
num_joints: 20

loader:
  train:
    batch_size: 4
    num_workers: 4
    shuffle: true
    pin_memory: true
  val:
    batch_size: 4
    num_workers: 2
    shuffle: false
    pin_memory: true
  test:
    batch_size: 4
    num_workers: 2
    shuffle: false
    pin_memory: true
```

### 3.3 モデル: `configs/models/tennis_mvpose.yaml`

`TennisDetrConfig` に対応:

```yaml
D_model: 256
dim_feedforward: 1024
nheads: 8
encoder_layers: 6
decoder_layers: 6
dropout: 0.1

num_joints: 20
num_court_points: 20
num_queries: 6
max_cameras: 8
max_frames: 32
```

### 3.4 トレーニング: `configs/training/tennis_mvpose.yaml`

```yaml
seed: 42

trainer:
  max_epochs: 50
  accelerator: auto
  devices: 1
  precision: 16-mixed
  gradient_clip_val: 1.0

optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4

max_steps: 0          # >0 の場合、CosineAnnealingLR(T_max=max_steps) を有効化

loss:
  lambda_pose: 1.0
  lambda_exist: 0.1
  lambda_vel: 0.0
```

### 3.5 ロギング: `configs/logging/tennis_mvpose.yaml`

```yaml
logger:
  save_dir: runs
  name: tennis_pose
  default_hp_metric: false

callbacks:
  checkpoint:
    monitor: val/total
    mode: min
    save_top_k: 1
    save_last: true
    filename: "{epoch:03d}-{val_total:.3f}"
  lr_monitor:
    logging_interval: epoch

visualizer:
  max_batches: 2
  exist_threshold: 0.5
```

---

## 4. LightningModule: `TennisDetrModule`

実装: `src/training/tennis/lightning.py:TennisDetrModule`

### 4.1 入出力

- `forward(batch)`:
  - 入力バッチは `TennisPoseDataModule` からの dict:
    - `"keypoints_2d": [B,T,V,M,J,2]`
    - `"player_mask": [B,T,V,M]`
    - `"court_2d": [B,V,20,2]`
    - `"pose_3d_gt": [B,T,M,J,3]`
    - `"exist_3d_gt": [B,T,M]`
  - `TennisDETR` に渡し、`{"pose_3d": [B,Q,T,J,3], "exist_conf": [B,Q,1]}` を返す。

### 4.2 損失計算（概要）

```python
pose_pred  # [B,Q,T,J,3]
exist_pred # [B,Q,1]
pose_gt    # [B,T,M,J,3]
exist_gt   # [B,T,M]
```

- 対応付け:
  - GT を `[B,M,T,J,3]` に permute してから `[B,Q,T,J,3]` にコピー（`Q >= M` 前提で `M` まで）。
  - `exist_gt.any(dim=1)` を `[B,M]` として、`[B,Q,1]` にコピー。
- 損失:
  - `pose_l1`: 存在 Query に対する L1（`pose_mask` でマスクし、非ゼロ要素数で割る）。
  - `exist_bce`: `binary_cross_entropy(exist_pred, exist_gt_query)`。
  - `vel_l2`: オプション。`pose_pred[:, :, 1:] - pose_pred[:, :, :-1]` の L2。
  - 合計: `total = λ_pose * pose_l1 + λ_exist * exist_bce + λ_vel * vel_l2`。

### 4.3 ロギングと可視化

- `training_step` / `validation_step` で `train/total`, `val/total` などを `self.log`。
- `validation_step` では、`visualizer.max_batches` までのバッチについて、簡易な GT vs Pred 2D オーバーレイ画像を TensorBoard に保存。
  - 実装イメージ:

    ```python
    if batch_idx < cfg.logging.visualizer.max_batches:
        img = _render_2d_overlay(batch, outputs)  # [3,H,W]
        self.logger.experiment.add_image(
            "val/pose2d_gt_vs_pred",
            img,
            global_step=self.global_step,
        )
    ```

  - 将来的には `src/visualize/tennis_pose.py` 由来の 3D/2D レンダリングユーティリティで置き換え可能。

---

## 5. CLI: `train_tennis_pose.py`

実装: `src/cli/train_tennis_pose.py`

- 役割:
  - `--config configs/tennis_pose.yaml` と任意の `--set key=value` から DictConfig を構築し、上記パイプラインを起動する。
- 主な引数:

| 引数 | 説明 |
| --- | --- |
| `--config` | トップレベル YAML (`configs/tennis_pose.yaml`) |
| `--set` | `dataset.name=... training.trainer.max_epochs=...` などの dotlist 形式オーバーライド |

- 内部処理:
  1. `load_cfg(config_path, overrides)` で設定を読み込む。
  2. `cfg.task == "tennis_pose"` を確認（異なる場合は使用法エラー）。
  3. `ConfigLoader(cfg)` を用いて DataModule, LightningModule, Logger, Callbacks, Trainer を構築。
  4. `trainer.fit(lit, datamodule=dm)` を起動。

---

## 6. 実行例

```bash
python src/cli/train_tennis_pose.py \
  --config configs/tennis_pose.yaml \
  --set dataset.name=sim_fps60_dur3p0_C4_P1-20_T10 \
        training.trainer.max_epochs=5
```

このコマンドにより、`data/tennis_autogen/sim_fps60_dur3p0_C4_P1-20_T10` を読み込み、`TennisDETR` を 5 エポック訓練する。TensorBoard ログは `runs/tennis_pose/<experiment_name>/` に保存される。
