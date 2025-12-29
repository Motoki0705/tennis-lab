# test_generate_dataset.py (PLCS)

データセット生成スクリプトのテストです。

**ファイル**: `tests/e2e/plcs/test_generate_dataset.py`

**対象スクリプト**: `src.plcs.scripts.generate_dataset`

## 概要

このテストファイルは以下を検証します：

1. スクリプトが正常終了すること
2. シーンファイルが生成されること
3. メタデータファイルが生成されること

---

## test_plcs_generate_dataset

標準設定でのデータセット生成テスト。

### 検証内容

1. スクリプトが `returncode == 0` で終了
2. 出力ディレクトリが作成される
3. `scenes/` ディレクトリが作成される
4. `scene_*.npz` ファイルが1つ以上存在（最大10）
5. `stats.json` が作成される
6. `scenes_meta.json` が作成される
7. `meta.json` が作成される

### 使用される設定オーバーライド

```bash
simulation.num_scenes=10
simulation.num_cameras=2
simulation.human_visibility_threshold=0.3
```

### タイムアウト

300秒

---

## test_plcs_generate_dataset_custom_settings

カスタム設定でのデータセット生成テスト。

### 検証内容

1. Hydra設定オーバーライドが機能
2. シーンファイルが1つ以上生成（最大5）

### 使用される設定オーバーライド

```bash
simulation.num_scenes=5
simulation.num_cameras=1
simulation.human_visibility_threshold=0.3
```

---

## 出力検証

### 期待されるディレクトリ構造

```
{output_dir}/
├── scenes/
│   ├── scene_000000.npz
│   ├── scene_000001.npz
│   └── ...
├── meta.json
├── stats.json
└── scenes_meta.json
```

### 注意事項

- 可視性フィルタリングにより、要求したシーン数より少ない場合がある
- `human_visibility_threshold` の設定に依存
