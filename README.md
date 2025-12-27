# Tennis-Lab 🎾

単眼動画からテニスシーンを3D再構築するためのモジュラーパイプライン

## 概要

Tennis-Labは、単眼カメラで撮影されたテニスの試合動画から、プレーヤーとボールの3D軌道を復元するための研究プロジェクトです。

### パイプライン全体像

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         入力: 単眼テニス動画                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │     GVHMR       │  │      WASB       │  │   Court KP      │
        │  (third_party)  │  │   (src/wasb)    │  │   Detection     │
        │                 │  │                 │  │                 │
        │ • SMPL推定      │  │ • ボール位置    │  │ • コートKP検出  │
        │ • ViTPose(2D)   │  │   (2D画像上)    │  │   (2D画像上)    │
        └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                 │                    │                    │
                 │ 2Dスケルトン       │ 2Dボール位置       │ 2Dコートキーポイント
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌─────────────────┐  ┌─────────────────┐
        │      PLCS       │  │      BLCS       │
        │   (src/plcs)    │  │   (src/blcs)    │
        │                 │  │                 │
        │ • 2Dスケルトン  │  │ • 2Dボール位置  │
        │   → 3D位置      │  │   → 3D軌道      │
        └────────┬────────┘  └────────┬────────┘
                 │                    │
                 ▼                    ▼
        ┌─────────────────────────────────────────┐
        │           3D Tennis Scene               │
        │                                         │
        │  • プレーヤー: コート上の3D位置 + SMPL  │
        │  • ボール: 3D軌道（バウンス検出含む）   │
        └─────────────────────────────────────────┘
```

---

## モジュール詳細

### GVHMR (third_party/)

**機能**: 動画からプレーヤーの3Dメッシュ（SMPL）を推定

- 入力: 単眼動画
- 出力:
  - SMPLパラメータ（ローカル空間）
  - 2Dスケルトン（ViTPoseによる副産物）

#### 可視化例

| 入力動画 | SMPL重畳出力 |
|:--------:|:------------:|
| ![入力動画](assets/gvhmr/input.mp4) | ![SMPL出力](assets/gvhmr/output.mp4) |

```bash
# GVHMRの実行
uv run python third_party/GVHMR/tools/demo/demo_multi.py --video inputs/demo/match.mp4
```

---

### WASB (src/wasb/)

**機能**: 画像上のボール位置を特定（Where is the Ball?）

- 入力: 単眼動画
- 出力: フレームごとのボール2D座標 (u, v)

#### 可視化例

| 入力動画 | ボール検出オーバーレイ |
|:--------:|:----------------------:|
| ![入力動画](assets/wasb/input.mp4) | ![ボール検出](assets/wasb/output.mp4) |

```bash
# WASBの実行（アンサンブル推論）
uv run python -m src.wasb.scripts.visualize.ball_video_ensemble \
    video_path=inputs/demo/match.mp4 \
    output_path=assets/wasb/output.mp4
```

---

### PLCS (src/plcs/)

**機能**: 2Dスケルトンからコート上の3Dプレーヤー位置を推定

- 入力:
  - 2D人物スケルトン（17キーポイント, COCO形式）
  - 2Dコートキーポイント（20点）
- 出力: コート座標系での3D位置 (x, y, z) + 向き (yaw)

#### 可視化例

| 2D入力シーン | 3D推論結果 |
|:------------:|:----------:|
| ![2D入力](assets/plcs/input.mp4) | ![3D出力](assets/plcs/output.mp4) |

```bash
# PLCSの可視化（入力シーン + 3D推論結果を保存）
uv run python -m src.plcs.scripts.visualize \
    visualization.mode=predict \
    visualization.view=animation \
    visualization.animation_view=3d \
    visualization.checkpoint=outputs/plcs/checkpoints/last.ckpt \
    visualization.save=assets/plcs/output.mp4 \
    visualization.save_input=assets/plcs/input.mp4
```

---

### BLCS (src/blcs/)

**機能**: 2Dボール位置からコート上の3Dボール軌道を推定

- 入力:
  - 2Dボール位置 (u, v)
  - 2Dコートキーポイント（20点）
- 出力: コート座標系での3Dボール軌道 (x, y, z)

#### 可視化例

| 2D入力シーン | GT vs 予測（3D比較） |
|:------------:|:--------------------:|
| ![2D入力](assets/blcs/input.mp4) | ![GT vs 予測](assets/blcs/output.mp4) |

> **Note**: 予測モードでは、**緑色がGT軌道**、**赤色が予測軌道**で描画されます。
> これによりモデルの精度と限界を直感的に把握できます。

```bash
# BLCSの可視化（入力シーン + GT vs 予測の比較アニメーションを保存）
uv run python -m src.blcs.scripts.visualize \
    visualization.mode=predict \
    visualization.view=animation \
    visualization.animation_view=3d \
    visualization.checkpoint=outputs/blcs/checkpoints/last.ckpt \
    visualization.save=assets/blcs/output.mp4 \
    visualization.save_input=assets/blcs/input.mp4
```

---

## セットアップ

```bash
# 依存関係のインストール
uv sync

# GVHMRのセットアップ（サブモジュール）
cd third_party/GVHMR
# ... GVHMRの依存関係をインストール
```

---

## ディレクトリ構造

```
tennis-lab/
├── src/
│   ├── base/           # 共通抽象化・ベースクラス
│   ├── blcs/           # Ball Localization in Court System
│   ├── plcs/           # Player Localization in Court System
│   ├── wasb/           # Where's the Ball (ボール検出)
│   └── utils/          # 共有ユーティリティ
├── third_party/
│   └── GVHMR/          # 3D人物メッシュ推定
├── data/               # データセット
├── outputs/            # 学習結果・チェックポイント
├── assets/             # README用可視化素材
└── configs/            # 設定ファイル
```

---

## ライセンス

（ライセンス情報をここに記載）

---

## 引用

（関連論文の引用情報をここに記載）
