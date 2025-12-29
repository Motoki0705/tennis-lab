# test_generate_dataset.py (BLCS)

データセット生成スクリプトのテストです。

**ファイル**: `tests/e2e/blcs/test_generate_dataset.py`

**対象スクリプト**: `src.blcs.scripts.generate_dataset`

## 概要

このテストファイルは以下を検証します：

1. スクリプトが正常終了すること
2. シーンファイルが生成されること
3. メタデータファイルが生成されること

---

## test_blcs_generate_dataset

標準設定でのデータセット生成テスト。

### 検証内容

1. スクリプトが `returncode == 0` で終了
2. 出力ディレクトリが作成される
3. `scenes/` ディレクトリが作成される
4. `scene_*.npz` ファイルが1つ以上存在
5. `meta.json` が作成される

### 使用される設定オーバーライド

```bash
sampling.per_from_cell_samples=10
generator.num_cameras_sampled=2
generator.max_attempts_per_cell=50
```

### タイムアウト

300秒

---

## test_blcs_generate_dataset_minimal_samples

最小設定での高速生成テスト。

### 検証内容

1. スクリプトが正常終了
2. シーンファイルが1つ以上生成

### 使用される設定オーバーライド

```bash
sampling.per_from_cell_samples=5
generator.num_cameras_sampled=1
generator.ball_visibility_threshold=0.5
generator.max_attempts_per_cell=50
```

### 目的

CI/CDでの高速テスト実行。最小限の設定でスクリプトが動作することを確認。

---

## 出力検証

### 期待されるディレクトリ構造

```
{output_dir}/
├── scenes/
│   ├── scene_000000.npz
│   ├── scene_000001.npz
│   └── ...
└── meta.json
```

### 注意事項

- 可視性フィルタリングにより、要求したシーン数より少ない場合がある
- `ball_visibility_threshold` の設定に依存
