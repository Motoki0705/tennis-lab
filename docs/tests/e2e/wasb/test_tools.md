# test_tools.py (WASB)

WASBツールスクリプトのテストです。

**ファイル**: `tests/e2e/wasb/test_tools.py`

## 概要

WASBのツールスクリプト（バックボーン抽出、パッチトークンエンコーディング）のテストです。

多くのテストは実際のチェックポイントが必要なためスキップされています。

---

## test_extract_dinov3_backbone

DinoV3バックボーン抽出テスト。

### 状態

**スキップ**: DinoV3ベースのチェックポイントが必要

### 対象スクリプト

`src.wasb.scripts.tools.extract_dinov3_backbone`

### 目的

学習済みモデルからDinoV3バックボーン部分のみを抽出。

### スキップ理由

実際のWASBチェックポイントには以下が必要：
- DinoV3バックボーンの重み
- 特定のモデルアーキテクチャ

モックチェックポイントでは代替不可。

---

## test_encode_dinov3_patch_tokens

DinoV3パッチトークンエンコーディングテスト。

### 状態

**スキップ**: WASBLightningModuleチェックポイントが必要

### 対象スクリプト

`src.wasb.scripts.tools.encode_dinov3_patch_tokens`

### 目的

データセット画像をDinoV3でエンコードし、パッチトークン埋め込みを保存。

### スキップ理由

実際のWASBチェックポイントが必要。

---

## test_encode_dinov3_patch_tokens_without_checkpoint

チェックポイントなしでのエンコーディングテスト。

### 状態

**実行可能**

### 対象スクリプト

`src.wasb.scripts.tools.encode_dinov3_patch_tokens`

### 検証内容

1. デフォルトモデルでスクリプトが起動可能
2. チェックポイント関連エラーの場合は許容

### 使用される設定

```bash
data.root_dir={dataset_dir}
output_dir={output_dir}
data.train_matches=[game1]
data.val_matches=[]
data.test_matches=[]
device=cpu
num_augments=1
```

### 備考

スクリプトがチェックポイントを必要とする場合はエラーとなるが、
これは想定内の動作として許容。

---

## 実際のテストを行う場合

実際のWASBモデルでテストするには：

1. 学習済みチェックポイントを用意
2. テストを `@pytest.mark.skip` から解除
3. チェックポイントパスを設定
