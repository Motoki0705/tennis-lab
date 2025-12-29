# test_train.py (BLCS)

学習スクリプトのテストです。

**ファイル**: `tests/e2e/blcs/test_train.py`

**対象スクリプト**: `src.blcs.scripts.train`

## 概要

このテストファイルは以下を検証します：

1. 学習スクリプトが正常終了すること
2. チェックポイントが生成されること
3. 設定ファイルが保存されること
4. ドライランモードが動作すること

---

## test_blcs_train_basic

基本学習テスト（GPU必須）。

### マーカー

```python
@pytest.mark.e2e
@pytest.mark.cuda
```

### 検証内容

1. スクリプトが正常終了
2. 出力ディレクトリが作成される
3. `checkpoints/` ディレクトリが作成される
4. `.ckpt` ファイルが1つ以上存在
5. `config.yaml` が作成される

### 使用される設定

```bash
training.max_epochs=1
run.gpus=1
run.fast_dev_run=true
data.batch_size=2
```

### 前提条件

- CUDA対応GPU
- `create_minimal_blcs_dataset` で生成したデータセット（10シーン）

---

## test_blcs_train_with_custom_params

カスタムパラメータでの学習テスト（GPU必須）。

### マーカー

```python
@pytest.mark.e2e
@pytest.mark.cuda
```

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

## test_blcs_train_dry_run

ドライランモードのテスト（CPUのみ）。

### マーカー

```python
@pytest.mark.e2e
# GPU不要
```

### 検証内容

1. ドライランモードが正常終了
2. 出力ディレクトリが作成される
3. `config.yaml` が作成される
4. 標準出力に "dry run" が含まれる

### 使用される設定

```bash
run.dry_run=true
data.batch_size=2
```

### ドライランの動作

- DataLoaderを初期化してバッチ情報を表示
- 実際の学習は実行しない
- チェックポイントは生成しない
- GPU不要

### 目的

- データローダーの設定が正しいことを確認
- CI/CDでGPUなしでもテスト可能

---

## 出力検証

### 期待されるディレクトリ構造（通常モード）

```
{output_dir}/
├── checkpoints/
│   └── *.ckpt
└── config.yaml
```

### 期待されるディレクトリ構造（ドライラン）

```
{output_dir}/
└── config.yaml
```
