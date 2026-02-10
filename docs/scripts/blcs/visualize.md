# BLCS visualize

BLCSシーンの可視化および学習済みモデルによる予測結果の確認を行うスクリプト。

## 概要

このスクリプトは、生成されたBLCSシーンデータの可視化や、学習済みモデルを使用した予測結果の可視化を行います。3Dビュー、2Dトップダウンビュー、カメラビュー、アニメーションなど、複数の表示モードをサポートします。

## コマンド例

```bash
# デフォルト設定で可視化
uv run python -m src.blcs.scripts.visualize

# シーンパスと情報表示を指定
uv run python -m src.blcs.scripts.visualize visualization.scene_path=data/blcs/scenes/scene_000000.npz visualization.info=true

# 特定のフレームとカメラを指定
uv run python -m src.blcs.scripts.visualize visualization.frame=10 visualization.camera=2

# 3Dビューを表示
uv run python -m src.blcs.scripts.visualize visualization.view=3d

# アニメーションを保存
uv run python -m src.blcs.scripts.visualize visualization.view=animation visualization.save=output.mp4

# 予測モードで実行
uv run python -m src.blcs.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/blcs/single/logs/version_0/checkpoints/last.ckpt

# 入力シーンアニメーションも保存
uv run python -m src.blcs.scripts.visualize visualization.save_input=input_anim.mp4
```

## コンフィグ

エントリポイント: `src/blcs/configs/visualize.yaml`

### defaults 構成

```yaml
defaults:
  - visualization: single
  - run: visualize
```

### visualization (可視化設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mode` | `visualize` | モード (visualize/predict) |
| `scene_path` | `data/blcs/scenes/scene_000003.npz` | シーンファイルパス |
| `frame` | `0` | 表示するフレームインデックス |
| `view` | `multi` | 表示モード |
| `camera` | `0` | カメラインデックス |
| `animation_view` | `2d` | アニメーションビュータイプ |
| `fps` | `null` | アニメーションFPS (nullでシーンのFPSを使用) |
| `save` | `null` | 出力ファイルパス |
| `save_input` | `null` | 入力シーンアニメーション保存パス |
| `info` | `false` | シーン情報を表示するか |
| `checkpoint` | `outputs/blcs/single/logs/version_0/checkpoints/last.ckpt` | 予測モード用チェックポイント |
| `output` | `null` | 予測結果出力パス |

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `device` | `auto` | デバイス (auto/cuda/cpu) |

### view オプション

| 値 | 説明 |
|----|------|
| `multi` | 複数ビューを同時表示 (3D + 2D + Camera) |
| `3d` | 3Dパースペクティブビュー（ボール軌道を表示） |
| `2d` | 2Dトップダウンビュー（コート俯瞰） |
| `camera` | カメラ視点ビュー（2D投影位置を表示） |
| `animation` | アニメーション表示/保存 |

### animation_view オプション

| 値 | 説明 |
|----|------|
| `2d` | 2Dトップダウンアニメーション |
| `3d` | 3Dアニメーション |
| `camera` | カメラビューアニメーション |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             visualize.py                                     │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  mode=visualize │                                                        │
│  │                 │                                                        │
│  │  load_scene()   │──▶ BLCSSceneRenderer ──▶ matplotlib 表示/保存         │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐                               │
│  │  mode=predict   │      │                 │                               │
│  │                 │      │  BLCSモデル     │                               │
│  │  load_scene()   │──▶   │  (checkpoint)   │──▶ 予測結果可視化             │
│  └─────────────────┘      └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー (visualize モード):
1. NPZ シーンファイルを読み込み
2. (info=true の場合) シーン情報を表示
3. 指定されたビューでレンダリング
   - 3D: ボール軌道 + コート + カメラ位置
   - 2D: トップダウンビューでボール軌道
   - camera: カメラ視点での2D投影
4. (save が指定された場合) ファイルに保存、なければ表示

処理フロー (predict モード):
1. NPZ シーンファイルを読み込み
2. チェックポイントからモデルをロード
3. 2D入力データで推論
4. Ground Truth と予測結果を比較可視化
```

## 表示内容

### 3D ビュー
- ボールの3D軌道（時間で色分け）
- テニスコートの3Dモデル
- カメラ位置（フラスタム表示）
- ネット

### 2D トップダウンビュー
- コート上面図
- ボール軌道の投影
- バウンス位置のマーク

### カメラビュー
- カメラ視点での2D投影位置
- コートのキーポイント

### 予測モード表示
- Ground Truth 軌道（青）
- 予測軌道（赤）
- 誤差の可視化

## 出力例

- 静止画: PNG ファイル
- アニメーション: MP4 ファイル

## 関連モジュール

- `src.blcs.generate_dataset.io.dataset_io`: シーン読み込み
- `src.utils.rendering.BLCSSceneRenderer`: シーンレンダリング
