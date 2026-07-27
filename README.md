# Tennis-Lab 🎾

単眼テニス動画から、プレーヤー/ボール/コートを推定して「コート座標系の3Dシーン」に統合するための研究用モジュラーパイプライン。

## 成果

### Tennis Scene（統合3D再構成）

| 入力映像（`data/samples/tennis_clip.mp4`） | コート座標系の3Dシーン |
| :---: | :---: |
| <img src="assets/tennis_scene/tennis_clip.gif" width="400" /> | <img src="assets/tennis_scene/rich_scene_rendering.gif" width="400" /> |

- 実装/実行: [src/tennis_scene/README.md](src/tennis_scene/README.md)

### Ball Detection（2Dボール検出）

<p align="center">
  <img src="assets/ball_detection/game1_clip1_prediction.gif" width="840" />
</p>

- 実装/実行: [src/tasks/ball_detection/README.md](src/tasks/ball_detection/README.md)

### Court Detection（KP / SEG / LINE）

<p align="center">
  <img src="assets/court_detection/kp.gif" width="840" /><br/>
  <em>Keypoint：RGB + 予測キーポイント ｜ 平均ヒートマップ</em>
</p>

<p align="center">
  <img src="assets/court_detection/seg.gif" width="840" /><br/>
  <em>Segmentation：RGB ｜ セグメンテーションマップ</em>
</p>

<p align="center">
  <img src="assets/court_detection/line.gif" width="840" /><br/>
  <em>Line：RGB ｜ ラインマップ</em>
</p>

- 実装: [src/tasks/court_detection/README.md](src/tasks/court_detection/README.md)

### PLCS（プレーヤー3D位置・yaw推定）

<p align="center">
  <img src="assets/plcs/gt_vs_pred.gif" width="720" /><br/>
  <em>GT（緑）｜ Prediction（赤）</em>
</p>

- 実装/実行: [src/tasks/plcs/README.md](src/tasks/plcs/README.md)

### BLCS（ボール3D軌道推定）

<p align="center">
  <img src="assets/blcs/gt_vs_pred.gif" width="600" /><br/>
  <em>GT（緑）｜ Prediction（赤）</em>
</p>

- 実装/実行: [src/tasks/blcs/README.md](src/tasks/blcs/README.md)

### BLCSデータ生成（物理シミュレーション）

<p align="center">
  <img src="assets/blcs_sim/blcs_sim_demo.png" width="840" /><br/>
  <em>物理シミュレーションによるボール軌道・イベント・マルチカメラ観測の生成</em>
</p>

## 開発中

### 3DGS Synthetic Data Generation（3DGSによる合成データ生成）

3D Gaussian Splatting (3DGS) で再構成された実写コート空間に対して、2面隣接コートの幾何モデル（`b00-ground-court-geometry-v1`）とカメラ座標アライメントを自動適合させ、幾何的に一貫した合成データを生成・拡張するパイプラインを開発しています。

<p align="center">
  <img src="assets/synthetic_data_generation/3dgs_court_alignment.gif" width="840" /><br/>
  <em>3DGS レンダリング ＆ 2面コート白線投影オーバーレイ（左）｜ 2面コート空間と SfM カメラ軌道の 3D 可視化（右）</em>
</p>

- **Dual-Court Alignment**: 隣接する2面のテニスコート（Court-0：シアン、Court-1：ゴールド）を同時に推定・保持し、任意のカメラ視点への正確な 2D 投影白線を生成
- **SfM Trajectory Navigation**: SfM により復元された 400 フレーム超のカメラ軌道と視線方向（FOV 矢印）を 3D コート空間上で追跡可視化
- **実装/検証**: `src/synthetic_data_generation` / `data/tennis/3dgs_alignment`

### Multi-object PLCS / BLCS

[PR #650](https://github.com/Motoki0705/tennis-lab/pull/650) で、複数プレーヤー・複数ボールの lifecycle を扱う生成・追跡・可視化パイプラインを開発しています。

| 複数プレーヤー（PLCS） | 複数ボール（BLCS） |
| :---: | :---: |
| <img src="assets/plcs/multi_object.gif" width="400" /> | <img src="assets/blcs/multi_object.gif" width="400" /> |
| `multi_object_lifecycle_v2 / scene_000040` | `multi_object_lifecycle_v4 / scene_000345` |


## ライセンス / 引用

このリポジトリは [MIT License](LICENSE) の下で公開されています。

## 構造（成果の裏側）

### どこに何があるか（タスク）

- Ball Detection: 画像上の2Dボール位置（`src/tasks/ball_detection`）
- Court Detection: 20点コートキーポイント（`src/tasks/court_detection`）
- PLCS: 2Dスケルトン → コート上3Dプレーヤー位置/yaw（`src/tasks/plcs`）
- BLCS: 2Dボール位置 → コート上3Dボール軌道（`src/tasks/blcs`）
- GVHMR: 画像列 -> 2Dスケルトン + SMPL (`third_party/GVHMR`)
- 統合: 上記をまとめて1本のパイプラインとして回す (`src/tennis_scene/README.md`)
- 合成データ生成: 再構成済み3D sceneへ物理軌道を合成して学習データを公開する (`src/synthetic_data_generation/README.md`)

### 典型データフロー

```
Video
  ├─ Court Detection  → court_kp (2D)
  ├─ Ball Detection   → ball_uv (2D)
  ├─ (optional) GVHMR → human_kp (2D) + SMPL (local)
  ├─ PLCS             → player_pos/yaw (3D on court)
  └─ BLCS             → ball_pos_world (3D on court)
```

### ディレクトリの役割

- `src/`: タスク実装（各タスクは `configs/` + `scripts/` + `training/` などを持つ）
- `third_party/`: 外部モジュール（例: GVHMR）。vendor codeは隔離
- `data/`: データセット/入力（大きなデータやモデルはコミットしない）
- `outputs/`: 学習ログ・チェックポイント・生成物（大きなartifactはコミットしない）
- `assets/`: README用の軽量デモ素材（GIF/PNGなど）
- `docs/`: 使い方/テスト/Dockerなどの補助ドキュメント
