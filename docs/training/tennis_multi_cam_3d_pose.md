# Tennis Pose Training パイプライン仕様（Spec）

本書は、テニス用マルチビュー 3D ポーズ推定の学習パイプライン（ConfigLoader, DataModule, LightningModule, CLI）の仕様をまとめる。

対象:
- `src/training/utils/config.py:ConfigLoader`（`task == "tennis_multi_cam_3d_pose"` 分岐）
- `src/training/tennis_multi_cam_3d_pose/datamodule.py:TennisPoseDataModule`
- `src/training/tennis_multi_cam_3d_pose/lightning.py:TennisDetrModule` (v1用)
- `src/training/tennis_multi_cam_3d_pose/lightning_v2.py:TennisDetrV2Module` (v2用)
- `src/training/tennis_multi_cam_3d_pose/lightning_v2_5.py:TennisDetrV25Module` (v2.5用)
- `src/training/tennis_multi_cam_3d_pose/lightning_v3.py:TennisDetrV3Module` (v3用)
- `src/cli/tennis_multi_cam_3d_pose/train.py` (v1用)
- `src/cli/tennis_multi_cam_3d_pose/train_v2.py` (v2/v2.5/v3共用)
- YAML 設定: `configs/tennis_multi_cam_3d_pose.yaml` (v1用), `configs/tennis_multi_cam_3d_pose_v2.yaml` (v2用), `configs/tennis_multi_cam_3d_pose_v2_5.yaml` (v2.5用), `configs/tennis_multi_cam_3d_pose_v3.yaml` (v3用)

> **注**: v1（元のモデル）、v2（階層エンコーダ + 分離出力）、v2.5（v2拡張版）、v3（track-aware版）の全バージョンに対応。

---

## 1. ワークフロー（概要）

### 1.1 v1モデルのワークフロー

```python
from src.training.utils.config import load_cfg, ConfigLoader

cfg     = load_cfg("configs/tennis_multi_cam_3d_pose.yaml", overrides)
loader  = ConfigLoader(cfg)
dm      = loader.build_datamodule()       # TennisPoseDataModule
lit     = loader.build_lit_module()       # TennisDetrModule (v1)
logger  = loader.build_logger()           # TensorBoardLogger
cbs     = loader.build_callbacks()        # Checkpoint, LR monitor など
trainer = loader.build_trainer(logger, cbs)

trainer.fit(lit, datamodule=dm)
```

### 1.2 v2モデルのワークフロー

```python
from src.training.utils.config import load_cfg, ConfigLoader

cfg     = load_cfg("configs/tennis_multi_cam_3d_pose_v2.yaml", overrides)
loader  = ConfigLoader(cfg)
dm      = loader.build_datamodule()       # TennisPoseDataModule (v2対応)
lit     = loader.build_lit_module()       # TennisDetrV2Module (v2)
logger  = loader.build_logger()           # TensorBoardLogger
cbs     = loader.build_callbacks()        # Checkpoint, LR monitor など
trainer = loader.build_trainer(logger, cbs)

trainer.fit(lit, datamodule=dm)
```

**重要**: `ConfigLoader.build_lit_module()` は `training._target_` に基づいて自動的にv1/v2を判定する。

### 1.3 v2.5モデルのワークフロー

```python
from src.training.utils.config import load_cfg, ConfigLoader

cfg     = load_cfg("configs/tennis_multi_cam_3d_pose_v2_5.yaml", overrides)
loader  = ConfigLoader(cfg)
dm      = loader.build_datamodule()       # TennisPoseDataModule (v2対応)
lit     = loader.build_lit_module()       # TennisDetrV25Module (v2.5)
logger  = loader.build_logger()           # TensorBoardLogger
cbs     = loader.build_callbacks()        # Checkpoint, LR monitor など
trainer = loader.build_trainer(logger, cbs)

trainer.fit(lit, datamodule=dm)
```

### 1.4 v3モデルのワークフロー

```python
from src.training.utils.config import load_cfg, ConfigLoader

cfg     = load_cfg("configs/tennis_multi_cam_3d_pose_v3.yaml", overrides)
loader  = ConfigLoader(cfg)
dm      = loader.build_datamodule()       # TennisPoseDataModule (v2対応)
lit     = loader.build_lit_module()       # TennisDetrV3Module (v3)
logger  = loader.build_logger()           # TensorBoardLogger
cbs     = loader.build_callbacks()        # Checkpoint, LR monitor など
trainer = loader.build_trainer(logger, cbs)

trainer.fit(lit, datamodule=dm)
```

**重要**: `ConfigLoader.build_lit_module()` は `experiment_name` に基づいて自動的にv1/v2/v2.5/v3を判定する。

CLI では以下がそれぞれのフローをラップする:
- `src/cli/tennis_multi_cam_3d_pose/train.py` (v1用)
- `src/cli/tennis_multi_cam_3d_pose/train_v2.py` (v2/v2.5/v3共用)

---

## 2. ConfigLoader の `tennis_multi_cam_3d_pose` 分岐

実装: `src/training/utils/config.py:ConfigLoader`

### 2.1 `build_datamodule`

```python
def build_datamodule(self) -> DancetrackDataModule | TennisPoseDataModule:
    task = self._task()
    dataset_cfg = self.cfg.get("dataset")
    debug_cfg = self.cfg.get("debug")
    if task == "tennis_multi_cam_3d_pose":
        from src.training.tennis_multi_cam_3d_pose.datamodule import TennisPoseDataModule
        return TennisPoseDataModule(dataset_cfg, debug_cfg)
    ...
```

### 2.2 `build_lit_module`

v1/v2/v2.5/v3を自動判定する実装:

```python
def build_lit_module(self) -> SceneModelLightningModule | TennisDetrModule | TennisDetrV2Module | TennisDetrV25Module | TennisDetrV3Module:
    task = self._task()
    experiment_name = str(self.cfg.get("experiment_name") or "").lower()
    if task == "tennis_multi_cam_3d_pose":
        if "v3" in experiment_name:
            from src.training.tennis_multi_cam_3d_pose.lightning_v3 import TennisDetrV3Module
            return TennisDetrV3Module(self.cfg)
        if "v2_5" in experiment_name:
            from src.training.tennis_multi_cam_3d_pose.lightning_v2_5 import TennisDetrV25Module
            return TennisDetrV25Module(self.cfg)
        if "v2" in experiment_name:
            from src.training.tennis_multi_cam_3d_pose.lightning_v2 import TennisDetrV2Module
            return TennisDetrV2Module(self.cfg)
        else:
            from src.training.tennis_multi_cam_3d_pose.lightning import TennisDetrModule
            return TennisDetrModule(self.cfg)
    ...
```

**判定ロジック**:
- `experiment_name` に `v3` を含む → v3
- `experiment_name` に `v2_5` を含む → v2.5
- `experiment_name` に `v2` を含む → v2
- それ以外 → v1

これにより、YAML設定だけでv1/v2/v2.5/v3を切り替え可能。

その他のメソッド（`build_logger`, `build_callbacks`, `build_trainer`）は SceneModel 用と共通で、`cfg.logging` / `cfg.training.trainer` を解釈して TensorBoardLogger / ModelCheckpoint / LRMonitor / Trainer を構築する。

---

## 3. YAML 設定構成

### 3.1 v1トップレベル: `configs/tennis_multi_cam_3d_pose.yaml`

```yaml
task: tennis_multi_cam_3d_pose
experiment_name: tennis_mvpose_dev

includes:
  dataset: configs/datasets/tennis_multi_cam_3d_pose_sim.yaml
  model: configs/models/tennis_mvpose.yaml
  training: configs/training/tennis_mvpose.yaml
  logging: configs/logging/tennis_mvpose.yaml
```

### 3.2 v2トップレベル: `configs/tennis_multi_cam_3d_pose_v2.yaml`

```yaml
task: tennis_multi_cam_3d_pose
experiment_name: tennis_mvpose_v2_dev

includes:
  dataset: configs/datasets/tennis_multi_cam_3d_pose_sim.yaml
  model: configs/models/tennis_mvpose_v2.yaml
  training: configs/training/tennis_mvpose_v2.yaml
  logging: configs/logging/tennis_mvpose.yaml
```

### 3.3 v2.5トップレベル: `configs/tennis_multi_cam_3d_pose_v2_5.yaml`

```yaml
task: tennis_multi_cam_3d_pose
experiment_name: tennis_mvpose_dev_v2_5

includes:
  dataset: configs/datasets/tennis_multi_cam_3d_pose_sim.yaml
  model: configs/models/tennis_mvpose_v2_5.yaml
  training: configs/training/tennis_mvpose_v2_5.yaml
  logging: configs/logging/tennis_mvpose.yaml
```

### 3.4 v3トップレベル: `configs/tennis_multi_cam_3d_pose_v3.yaml`

```yaml
task: tennis_multi_cam_3d_pose
experiment_name: tennis_mvpose_dev_v3

includes:
  dataset: configs/datasets/tennis_multi_cam_3d_pose_sim.yaml
  model: configs/models/tennis_mvpose_v3.yaml
  training: configs/training/tennis_mvpose_v3.yaml
  logging: configs/logging/tennis_mvpose.yaml
```

**重要な違い**:
- 各バージョンで `model` と `training` がバージョン専用設定を指す
- `experiment_name` で v1/v2/v2.5/v3 を自動判定

- `load_cfg` が `includes` を展開し、`cfg.dataset`, `cfg.model`, `cfg.training`, `cfg.logging` を構成する。

### 3.5 データセット: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`

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

### 3.6 v1モデル: `configs/models/tennis_mvpose.yaml`

`TennisDetrConfig` に対応:

### 3.10 v1トレーニング: `configs/training/tennis_mvpose.yaml`

### 3.7 v2モデル: `configs/models/tennis_mvpose_v2.yaml`

`TennisDetrV2Config` に対応:

```yaml
_target_: src.models.tennis_multi_cam_3d_pose.TennisDetrV2Config
D_model: 256
dim_feedforward: 1024
nheads: 8
decoder_layers: 6
dropout: 0.1

# v2階層エンコーダパラメータ
intra_layers: 3
inter_layers: 3
temporal_layers: 3

num_joints: 20
num_court_points: 20
num_queries: 20
```

### 3.8 v2.5モデル: `configs/models/tennis_mvpose_v2_5.yaml`

`TennisDetrV2Config` を再利用（v2と同じ）:

```yaml
D_model: 256
dim_feedforward: 1024
nheads: 8
decoder_layers: 6
dropout: 0.1

intra_layers: 3
inter_layers: 3
temporal_layers: 3

num_joints: 20
num_court_points: 20
num_queries: 20
max_cameras: 8
max_frames: 32
```

### 3.9 v3モデル: `configs/models/tennis_mvpose_v3.yaml`

`TennisDetrV3Config` に対応:

```yaml
D_model: 256
dim_feedforward: 1024
nheads: 8
decoder_layers: 6
dropout: 0.1

intra_layers: 3
inter_layers: 3
temporal_layers: 3

num_joints: 20
num_court_points: 20
num_queries: 50
max_cameras: 8
max_frames: 32
```

```yaml
_target_: src.training.tennis_multi_cam_3d_pose.TennisDetrModule
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 10
  gamma: 0.5
loss:
  lambda_pose: 1.0
  lambda_exist: 0.05
trainer:
  max_epochs: 50
  accelerator: gpu
  devices: 1
```

```yaml
_target_: src.models.tennis_multi_cam_3d_pose.TennisDetrConfig
D_model: 256
dim_feedforward: 1024
nheads: 8
encoder_layers: 6
decoder_layers: 6
dropout: 0.1
num_joints: 20
num_court_points: 20
num_queries: 20
```

### 3.11 v2トレーニング: `configs/training/tennis_mvpose_v2.yaml`

### 3.12 v2.5トレーニング: `configs/training/tennis_mvpose_v2_5.yaml`

v2と同じ損失設定:

```yaml
seed: 42

trainer:
  max_epochs: 500
  accelerator: auto
  devices: 1
  precision: bf16-mixed
  gradient_clip_val: 1.0

optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4

max_steps: 7000

scheduler:
  name: cosine_with_warmup
  warmup_steps: 500
  min_lr_ratio: 0.1

loss:
  lambda_canonical: 1.0
  lambda_root_trans: 1.0
  lambda_root_rot: 0.5
  lambda_global: 1.0
  lambda_exist: 0.05
  lambda_vel: 0.0
  lambda_pose_match: 1.0
  lambda_exist_match: 0.05
```

### 3.13 v3トレーニング: `configs/training/tennis_mvpose_v3.yaml`

v2とほぼ同じ損失設定（v3用のデフォルト）:

```yaml
seed: 42

trainer:
  max_epochs: 500
  accelerator: auto
  devices: 1
  precision: bf16-mixed
  gradient_clip_val: 1.0

optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4

max_steps: 7000

scheduler:
  name: cosine_with_warmup
  warmup_steps: 500
  min_lr_ratio: 0.1

loss:
  lambda_canonical: 1.0
  lambda_root_trans: 1.0
  lambda_root_rot: 0.5
  lambda_global: 1.0
  lambda_exist: 0.5
  lambda_vel: 0.0
  lambda_pose_match: 1.0
  lambda_exist_match: 0.5
```

```yaml
_target_: src.training.tennis_multi_cam_3d_pose.TennisDetrV2Module
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 10
  gamma: 0.5
loss:
  lambda_canonical: 1.0
  lambda_root_trans: 1.0
  lambda_root_rot: 0.5
  lambda_global: 1.0
  lambda_exist: 0.05
  lambda_pose_match: 1.0
  lambda_exist_match: 0.05
trainer:
  max_epochs: 50
  accelerator: gpu
  devices: 1
```

**重要な違い**:
- v2では `_target_` が `TennisDetrV2Module` を指す（旧方式）
- v2.5/v3では `_target_` を使わず、`experiment_name` で判定（新方式）
- v2用の4要素損失設定が追加されている

### 3.14 ロギング: `configs/logging/tennis_mvpose.yaml`

v1/v2/v2.5/v3で共通のロギング設定:

```yaml
logger:
  save_dir: runs
  name: tennis_multi_cam_3d_pose
  default_hp_metric: false
```

---

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

## 4. LightningModule

### 4.1 v1: `TennisDetrModule`

実装: `src/training/tennis_multi_cam_3d_pose/lightning.py:TennisDetrModule`

#### 4.1.1 入出力

- `forward(batch)`:
  - 入力バッチは `TennisPoseDataModule` からの dict:
    - `"keypoints_2d": [B,T,V,M,J,2]`
    - `"player_mask": [B,T,V,M]`
    - `"court_2d": [B,V,20,2]`
    - `"pose_3d_gt": [B,T,M,J,3]`
    - `"exist_3d_gt": [B,T,M]`
  - `TennisDETR` に渡し、`{"pose_3d": [B,Q,T,J,3], "exist_conf": [B,Q,1]}` を返す。

#### 4.1.2 損失計算（概要）

```python
pose_pred  # [B,Q,T,J,3] - モデル出力
pose_gt    # [B,T,M,J,3] - GTデータ
```

### 4.2 v2.5: `TennisDetrV25Module`

実装: `src/training/tennis_multi_cam_3d_pose/lightning_v2_5.py:TennisDetrV25Module`

- v2と同じ損失・I/O・可視化ロジック
- モデルのみ `TennisDETR_v2_5` に変更

### 4.3 v3: `TennisDetrV3Module`

実装: `src/training/tennis_multi_cam_3d_pose/lightning_v3.py:TennisDetrV3Module`

- v2と同じ損失・I/O・可視化ロジック
- モデルのみ `TennisDETR_v3` に変更（track-aware temporal encoder 付き）

実装: `src/training/tennis_multi_cam_3d_pose/lightning_v2.py:TennisDetrV2Module`

#### 4.2.1 入出力

- `forward(batch)`:
  - 入力バッチはv1と同じ（v2用GTデータも含む）
  - `TennisDETR_v2` に渡し、以下を返す:
    ```python
    {
        "canonical_pose": [B,Q,T,J,3],
        "root_trans": [B,Q,T,3],
        "root_rot": [B,Q,T,2],
        "global_pose": [B,Q,T,J,3],
        "exist_conf": [B,Q,1]
    }
    ```

#### 4.2.2 損失計算（概要）

```python
canonical_pred  # [B,Q,T,J,3] - canonical pose
root_trans_pred # [B,Q,T,3]   - root translation
root_rot_pred   # [B,Q,T,2]   - root rotation
global_pred     # [B,Q,T,J,3] - global pose

# v2用GTデータとの損失計算
canonical_gt, root_trans_gt, root_rot_gt, global_gt
```

---

## 5. CLI

### 5.1 v1 CLI: `src/cli/tennis_multi_cam_3d_pose/train.py`

- 役割:
  - `--config configs/tennis_multi_cam_3d_pose.yaml` と任意の `--set key=value` から DictConfig を構築し、v1パイプラインを起動する。
- 主な引数:

| 引数 | 説明 |
| --- | --- |
| `--config` | トップレベル YAML (`configs/tennis_multi_cam_3d_pose.yaml`) |
| `--set` | `dataset.name=... training.trainer.max_epochs=...` などの dotlist 形式オーバーライド |

### 5.2 v2 CLI: `src/cli/tennis_multi_cam_3d_pose/train_v2.py`

- 役割:
  - `--config configs/tennis_multi_cam_3d_pose_v2.yaml` / `v2_5.yaml` / `v3.yaml` と任意の `--set key=value` から DictConfig を構築し、v2/v2.5/v3パイプラインを起動する。
- 主な引数:

| 引数 | 説明 |
| --- | --- |
| `--config` | トップレベル YAML（v2/v2.5/v3） |
| `--set` | `dataset.name=... training.trainer.max_epochs=...` などの dotlist 形式オーバーライド |

**内部処理**（v1/v2/v2.5/v3共通）:
1. `load_cfg(config_path, overrides)` で設定を読み込む。
2. `cfg.task == "tennis_multi_cam_3d_pose"` を確認（異なる場合は使用法エラー）。
3. `ConfigLoader(cfg)` を用いて DataModule, LightningModule, Logger, Callbacks, Trainer を構築。
4. `trainer.fit(lit, datamodule=dm)` を起動。

---

## 6. 実行例

### 6.1 v1モデルの学習

```bash
python src/cli/tennis_multi_cam_3d_pose/train.py \
  --config configs/tennis_multi_cam_3d_pose.yaml \
  --set dataset.name=sim_fps60_dur3p0_C4_P1-20_T10 \
  --set training.trainer.max_epochs=50
```

### 6.2 v2モデルの学習

```bash
python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v2.yaml \
  --set dataset.name=sim_fps60_dur3p0_C4_P1-20_T10 \
  --set training.trainer.max_epochs=50 \
  --set model.cfg.intra_layers=4
```

### 6.3 v2.5モデルの学習

```bash
python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v2_5.yaml \
  --set dataset.name=sim_fps60_dur3p0_C4_P1-20_T10 \
  --set training.trainer.max_epochs=50
```

### 6.4 v3モデルの学習

```bash
python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v3.yaml \
  --set dataset.name=sim_fps60_dur3p0_C4_P1-20_T10 \
  --set training.trainer.max_epochs=50 \
  --set model.num_queries=30
```

### 6.5 scriptsラッパの使用

```bash
# v1
./scripts/train/run_train_tennis_multi_cam_3d_pose.sh

# v2
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh

# v2.5
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2_5.sh

# v3
./scripts/train/run_train_tennis_multi_cam_3d_pose_v3.sh
```

---

## 7. v1/v2/v2.5/v3の比較

| 項目 | v1 | v2 | v2.5 | v3 |
|------|----|----|------|----|
| モデルアーキテクチャ | 単一エンコーダ | 階層エンコーダ | 階層エンコーダ + カメラ/時間埋め込み | 階層エンコーダ + track-aware temporal encoder |
| 出力形式 | pose_3d [B,Q,T,J,3] | 分離出力（4要素） | 分離出力（4要素） | 分離出力（4要素） |
| LightningModule | TennisDetrModule | TennisDetrV2Module | TennisDetrV25Module | TennisDetrV3Module |
| 設定ファイル | tennis_mvpose.yaml | tennis_mvpose_v2.yaml | tennis_mvpose_v2_5.yaml | tennis_mvpose_v3.yaml |
| 損失関数 | 単一ポーズ損失 | 4要素損失 | 4要素損失 | 4要素損失 |
| CLI | train.py | train_v2.py | train_v2.py | train_v2.py |
| データセット互換性 | 既存データ | 既存データ（自動GT生成） | 既存データ | 既存データ |
| 主な差分 | - | 階層エンコーダ | カメラ・時間埋め込みを明示的に付与 | Query ごとの時間軸 TransformerEncoder で track-aware |
