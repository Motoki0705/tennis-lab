# test_visualize.py (WASB)

WASBの可視化スクリプトのテストです。

**ファイル**: `tests/e2e/wasb/test_visualize.py`

## 概要

WASB（Where's the Ball）タスクの可視化スクリプト（`visualize_trajectory`）のE2Eテストです。

---

## test_visualize_trajectory

軌道可視化のスモークテスト。

### 状態

**スキップ**: 実際のチェックポイントが必要

### マーカー

- `@pytest.mark.e2e`

### 対象スクリプト

`src.wasb.scripts.visualize_trajectory`

### 目的

モデルが予測したボール軌道を可視化。

### スキップ理由

1. 実際の学習済みモデルチェックポイントが必要
2. モックチェックポイントでは推論パイプライン全体を実行できない

---

## 実際のテストを行う場合

可視化テストを実際に行うには：

1. 学習済みチェックポイントを用意
   - `test_train_ball_detection` で生成可能
2. テストから `@pytest.mark.skip` を除去
3. チェックポイントパスを設定

```bash
uv run pytest tests/e2e/wasb/test_visualize.py::test_visualize_trajectory -v \
  --checkpoint-path /path/to/checkpoint.ckpt
```

---

## 関連スクリプト

- `src.wasb.scripts.visualize_trajectory`: 軌道可視化
- 将来的に追加される可能性のある可視化スクリプト:
  - ヒートマップ可視化
  - イベント検出結果の可視化
