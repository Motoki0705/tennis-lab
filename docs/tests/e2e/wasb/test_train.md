# test_train.py (WASB)

WASBモジュールの学習テストです。

**ファイル**: `tests/e2e/wasb/test_train.py`

## 概要

WASB（Where's the Ball）タスクの各サブモジュールの学習パイプラインをE2Eでテストします。

- Ball Detection（ボール検出）
- Trajectory Completion（軌道補完）
- Event Detection（イベント検出）

---

## test_train_ball_detection

ボール検出モデルの学習テスト。

### 状態

**実行可能**（E2E）

### マーカー

- `@pytest.mark.e2e`

### 対象スクリプト

`src.wasb.scripts.train_ball_detection`

### 検証内容

1. 学習ループが正常に実行される
2. 損失が減少する
3. チェックポイントが保存される

### 使用される設定

```yaml
model:
  name: simple_cnn_detector

trainer:
  max_epochs: 2
  accelerator: cpu

data:
  root_dir: {sample_dataset}
  train_matches: [game1]
  val_matches: []
  test_matches: []
```

---

## test_train_ball_detection_cuda

GPUを使用したボール検出学習テスト。

### 状態

**条件付き実行**（CUDA必須）

### マーカー

- `@pytest.mark.e2e`
- `@pytest.mark.cuda`

### 対象スクリプト

`src.wasb.scripts.train_ball_detection`

### 検証内容

- CUDAアクセラレータでの学習実行

### GPU要件

- CUDAが利用可能であること

---

## test_train_trajectory

軌道補完モデルの学習テスト。

### 状態

**スキップ**: TrajectoryLightningModule未実装

### 対象スクリプト

`src.wasb.scripts.train_trajectory`

### 目的

ボールの軌道を補完・予測するモデルの学習。

### スキップ理由

軌道補完のLightningModule実装が未完成。

---

## test_train_event_detection

イベント検出モデルの学習テスト。

### 状態

**スキップ**: EventDetectionLightningModule未実装

### 対象スクリプト

`src.wasb.scripts.train_event_detection`

### 目的

テニスのイベント（バウンス、ヒット等）を検出するモデルの学習。

### スキップ理由

イベント検出のLightningModule実装が未完成。

---

## テスト実行例

```bash
# ボール検出学習（CPU）
uv run pytest tests/e2e/wasb/test_train.py::test_train_ball_detection -v

# ボール検出学習（GPU）
uv run pytest tests/e2e/wasb/test_train.py::test_train_ball_detection_cuda -v
```

---

## 関連フィクスチャ

### wasb_sample_dataset

サンプルデータセットパスを提供（`tests/e2e/fixtures/wasb_fixtures.py`）。

---

## チェックポイント検証

学習後、以下の形式でチェックポイントが保存されることを確認：

```
{output_dir}/
├── checkpoints/
│   ├── epoch=*.ckpt
│   └── last.ckpt
└── lightning_logs/
```
