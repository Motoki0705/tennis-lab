# PLCS Multi-View Train E2E Test

## 概要

`test_train_multiview.py` は、PLCSマルチビュー学習スクリプト (`src.plcs.scripts.train --config-name train_multiview`) のE2Eテストです。

## テスト対象

| テスト関数 | 説明 | マーカー |
|-----------|------|---------|
| `test_plcs_train_multiview_dry_run` | Dry Runモードでのデータローディング確認 | `@e2e` |
| `test_plcs_train_multiview_gpu` | GPU使用時の学習（fast_dev_run） | `@e2e`, `@cuda` |

## テスト内容

### test_plcs_train_multiview_dry_run

**目的**: マルチビューデータローダーが正しく機能することを確認

**検証項目**:
- スクリプトがエラーなく終了すること
- マルチビューデータのバッチ形式が正しいこと

**実行条件**: CPU環境でも動作

### test_plcs_train_multiview_gpu

**目的**: GPU環境での学習パイプラインが正常に動作することを確認

**検証項目**:
- スクリプトがエラーなく終了すること
- 出力ディレクトリが生成されること
- 設定ファイルが保存されること

**実行条件**: CUDA対応GPUが必要

## フィクスチャ

テストでは `make_multiview_plcs_scene()` と `create_minimal_multiview_plcs_dataset()` を使用して、
複数カメラを持つ最小限のテストデータセットを生成します。

### make_multiview_plcs_scene

```python
def make_multiview_plcs_scene(
    *, scene_id: str = "scene_000000", num_cameras: int = 3
) -> SceneData:
    """複数カメラを持つ最小限のPLCSシーンを作成."""
```

### create_minimal_multiview_plcs_dataset

```python
def create_minimal_multiview_plcs_dataset(
    output_dir: Path,
    num_scenes: int = 5,
    num_cameras: int = 3,
) -> None:
    """マルチビューPLCSデータセットを生成."""
```

## 実行方法

```bash
# マルチビュー学習テストのみ
uv run pytest tests/e2e/plcs/test_train_multiview.py -v

# Dry Runテストのみ（GPUなし環境）
uv run pytest tests/e2e/plcs/test_train_multiview.py::test_plcs_train_multiview_dry_run -v

# GPUテストのみ
uv run pytest tests/e2e/plcs/test_train_multiview.py::test_plcs_train_multiview_gpu -v -m cuda
```
