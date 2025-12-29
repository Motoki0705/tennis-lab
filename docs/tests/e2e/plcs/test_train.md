# test_train.py (PLCS)

学習スクリプトのテストです。

**ファイル**: `tests/e2e/plcs/test_train.py`

**対象スクリプト**: 
- `src.plcs.scripts.train` (フレームモデル)
- `src.plcs.scripts.train_sequence` (シーケンスモデル)

## 概要

このテストファイルは以下を検証します：

1. フレームモデル/シーケンスモデルの学習が正常終了
2. チェックポイントが生成されること
3. ドライランモードが動作すること

---

## test_plcs_train_basic

フレームモデルの基本学習テスト（GPU必須）。

### マーカー

```python
@pytest.mark.e2e
@pytest.mark.cuda
```

### 検証内容

1. スクリプトが正常終了
2. `checkpoints/` ディレクトリが作成される
3. `.ckpt` ファイルが1つ以上存在
4. `config.yaml` が作成される

### 使用される設定

```bash
training.max_epochs=1
run.gpus=1
run.fast_dev_run=true
data.batch_size=2
```

---

## test_plcs_train_sequence

シーケンスモデルの学習テスト（GPU必須）。

### マーカー

```python
@pytest.mark.e2e
@pytest.mark.cuda
```

### 対象スクリプト

`src.plcs.scripts.train_sequence`

### 検証内容

1. シーケンスモデル学習が正常終了
2. チェックポイントが生成される

### 使用される設定

```bash
training.max_epochs=1
run.gpus=1
run.fast_dev_run=true
data.batch_size=2
```

---

## test_plcs_train_with_custom_params

カスタムパラメータでの学習テスト（GPU必須）。

### 検証内容

Hydra設定オーバーライドが正しく機能することを確認。

### 使用される設定

```bash
training.max_epochs=2
run.gpus=1
run.fast_dev_run=true
data.batch_size=2
```

---

## test_plcs_train_dry_run

フレームモデルのドライランテスト（CPUのみ）。

### マーカー

```python
@pytest.mark.e2e
# GPU不要
```

### 検証内容

1. ドライランモードが正常終了
2. `config.yaml` が作成される
3. 標準出力に "dry run" が含まれる

### 使用される設定

```bash
run.dry_run=true
data.batch_size=2
```

---

## test_plcs_train_sequence_dry_run

シーケンスモデルのドライランテスト（CPUのみ）。

### 対象スクリプト

`src.plcs.scripts.train_sequence`

### 検証内容

1. シーケンスモデルのドライランが正常終了
2. `config.yaml` が作成される
3. 標準出力に "dry run" が含まれる

---

## モデル比較

| スクリプト | モデル | 入力 | 出力 |
|-----------|--------|------|------|
| `train` | `PLCSModel` | 単一フレーム | 位置(3) + 回転(2) |
| `train_sequence` | `PLCSSequenceModel` | 時系列フレーム | 時系列位置 + 回転 |
