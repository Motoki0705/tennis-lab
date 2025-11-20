# scripts: visualization

可視化系のスクリプトは、主に次の 2 つで構成される。

- `scripts/tools/render_tennis_augmented.sh`
- `scripts/vis/render_c3d_markers.py`

## 1. render_tennis_augmented.sh

- **役割**: `TennisSceneWindowDataset` にデータ拡張を適用したサンプルを動画としてレンダリングし、ざっと目視確認できるようにする。
- **内部で呼び出す CLI**: `src/datasets/tennis/tools/render_tennis_augmented.py`

### 1.1 使い方

```bash
./scripts/tools/render_tennis_augmented.sh
```

- 既定のデータセット設定: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`
- 追加のオプション（サンプル数や split など）は、そのまま引数として渡す:

```bash
./scripts/tools/render_tennis_augmented.sh --num-samples 8 --split val
```

## 2. render_c3d_markers.py

- **役割**: 3DTennisDS の C3D マーカーを 3D 点群／スケルトンとしてレンダリングし、動画として保存する。
- **実装**: `scripts/vis/render_c3d_markers.py`

### 2.1 使い方

```bash
uv run python scripts/vis/render_c3d_markers.py \
  data/raw/3dtennisds/tp1/bh/tp1_bh_s1.c3d \
  --mode skeleton \
  --out outputs/vis/tp1_bh_s1_skeleton.mp4 \
  --fps 50 \
  --stride 2
```

C3D のマーカー名やスケルトン定義の詳細は、スクリプト内のコメントを参照。
